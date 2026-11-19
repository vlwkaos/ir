#!/usr/bin/env bash
# Steps of .github/workflows/eval-matrix.yml: a bench.sh quality eval with its queries sharded over
# GitHub x64 runners (research/eval-compute.md).
#
# Usage (W = work dir; the workflow caches W/input/models, W/input/data and W/llm-cache):
#   eval-matrix.sh keys    <dataset> <bundle-dir> <W>     cache keys of the inputs (Kaggle auth)
#   eval-matrix.sh seed    <dataset> <bundle-dir> <W>     fill W/input from Kaggle; INDEX_KEY gate
#   eval-matrix.sh shard   <i/N> <mode> <bundle-dir> <W>  shard i of N through scripts/kaggle/run.py
#   eval-matrix.sh collect <W> <out-dir>                  the shard's partials, llm_cache and logs
#   eval-matrix.sh merge   <mode> <W> <out-dir> <shard-out-dir>...   llm_cache union, then metrics
#
# Inputs are the Kaggle datasets of scripts/kaggle/ci.sh (ir-models, ir-corpus-<ds>,
# ir-lindera-<alias>, ir-index-<ds>); the bundle is scripts/kaggle/build.sh output. keys and seed
# write $GITHUB_OUTPUT; shard, collect and merge read DATASET, GIT7 and RERANK_KEY from the env.
#
# RERANK_KEY = sha256 of the reranker GGUF, src/llm/scoring.rs and src/llm/reranker.rs: the
# llm_cache key in src/search/hybrid.rs (rerank) has only the GGUF file name, query and content
# hash. llm_cache rows move between runs only under the same RERANK_KEY (rerank_cache).
#
# Reranker evaluation (research/reranker-eval.md): mode=dump is a hybrid run that also writes each
# tier-2 window with its pointwise scores and costs; merge leaves it in W/dump (actions/cache under
# DUMP_KEY). mode=rerank ranks W/dump with the CONFIGS names of scripts/rerank-configs.json.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PY="$REPO_ROOT/scripts/kaggle/run.py"
BEIR="$REPO_ROOT/scripts/beir-eval.py"
RR_CONFIGS="$REPO_ROOT/scripts/rerank-configs.json"
W=""
# shellcheck source=scripts/kaggle/lib.sh
source "$REPO_ROOT/scripts/kaggle/lib.sh"

die() { echo "eval-matrix: $*" >&2; exit 1; }
# need_env: SIZE PRE LABEL of $DATASET, and the seed outputs.
need_env() {
    : "${DATASET:?}" "${GIT7:?}" "${RERANK_KEY:?}"
    dataset_spec "$DATASET" || exit 1
}

# emit <NAME> <value>: set it here and, as <name>, in $GITHUB_OUTPUT for the later jobs.
emit() {
    export "$1=$2"
    _log "$1=$2"
    [[ -z "${GITHUB_OUTPUT:-}" ]] || printf '%s=%s\n' "$(tr '[:upper:]' '[:lower:]' <<<"$1")" "$2" >> "$GITHUB_OUTPUT"
}

# rerank_cache <dst> [<src>...]: dst keeps only llm_cache rows stamped with RERANK_KEY (table
# bench_rerank_key); rows of a dst with another stamp or none are deleted first. Each src
# (a collection DB or an earlier dst) adds its rows when its stamp equals RERANK_KEY.
rerank_cache() {
    python3 - "$RERANK_KEY" "$@" <<'PY'
import os, sqlite3, sys
key, dst, *srcs = sys.argv[1:]
KT = "bench_rerank_key"
c = sqlite3.connect(dst, isolation_level=None)

def has(schema, table):
    return c.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE name = ?", (table,)).fetchone() is not None

def stamp(schema):
    row = c.execute(f"SELECT key FROM {schema}.{KT}").fetchone() if has(schema, KT) else None
    return row[0] if row else None

if stamp("main") != key:
    if has("main", "llm_cache"):
        n = c.execute("DELETE FROM llm_cache").rowcount
        print(f"rerank cache {dst}: dropped {n} rows (stamp {stamp('main') or 'none'})")
    c.execute(f"CREATE TABLE IF NOT EXISTS {KT} (key TEXT NOT NULL)")
    c.execute(f"DELETE FROM {KT}")
    c.execute(f"INSERT INTO {KT} VALUES (?)", (key,))
added = 0
for src in srcs:
    if not os.path.isfile(src):
        print(f"rerank cache: no {src}")
        continue
    c.execute("ATTACH DATABASE ? AS src", (src,))
    s = stamp("src")
    if s != key:
        print(f"rerank cache: {src} has stamp {s or 'none'}, not {key[:12]}: not used")
    elif has("src", "llm_cache"):
        if not has("main", "llm_cache"):
            c.execute(c.execute("SELECT sql FROM src.sqlite_master WHERE name = 'llm_cache'").fetchone()[0])
        added += c.execute("INSERT OR IGNORE INTO main.llm_cache SELECT * FROM src.llm_cache").rowcount
    c.execute("DETACH DATABASE src")
total = c.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] if has("main", "llm_cache") else 0
# ^ bench.sh load_index copies only <collection>.sqlite: fold the WAL in before it does.
c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
c.close()
print(f"rerank cache {dst}: {total} rows, added {added} (key {key[:12]})")
PY
}

# rr_json models [<name>...]   "<file> <sha256> <url>" lines of the models the named configs use
# rr_json configs [<name>...]  JSON list of the named configs, each with "name"
# No names: the "dump" list of scripts/rerank-configs.json.
rr_json() {
    python3 - "$RR_CONFIGS" "$@" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
what, names = sys.argv[2], sys.argv[3:] or spec["dump"]
unknown = [n for n in names if n not in spec["configs"]]
if unknown:
    sys.exit(f"rerank configs: unknown {unknown} (known: {', '.join(spec['configs'])})")
configs = [{"name": n, **spec["configs"][n]} for n in names]
if what == "models":
    # ^ Only these: every model in one cache would pass the 10 GB actions cache limit.
    for f in sorted({c.get("env", {}).get("IR_LISTWISE_MODEL") for c in configs} - {None}):
        m = spec["models"][f]
        print(f, m["sha256"], m["url"])
else:
    print(json.dumps(configs))
PY
}

# fetch_models <dir>: the listwise GGUFs of CONFIGS from their pinned Hugging Face URL, sha256
# checked. Not in actions/cache: 20 shards restoring GBs each pass the account's cache egress limit.
fetch_models() {
    local dir="$1" f want url
    mkdir -p "$dir"
    # shellcheck disable=SC2086  # ^ CONFIGS is a space-separated list of names
    while read -r f want url; do
        [[ ! -f "$dir/$f" ]] || continue
        _log "downloading $f"
        curl -fsSL --retry 5 --retry-all-errors -o "$dir/$f.part" "$url" || die "download $url"
        [[ "$(sha "$dir/$f.part")" == "$want" ]] || die "$f: sha256 is not $want"
        mv "$dir/$f.part" "$dir/$f"
    done < <(rr_json models $CONFIGS)
}

# src_cat <pathspec>...: the bytes of the tracked files, in git order, for a cache key.
src_cat() { (cd "$REPO_ROOT" && git ls-files -z -- "$@" | xargs -0 cat); }

# rr_saved <i/N>: where W/llm-cache keeps a finished rerank shard's result partials, keyed by the
# dump, the configs with their models and the code that ranks them. merge carries every shard's
# into the merged cache, so a new dispatch runs only the shards that have none.
rr_saved() {
    local key
    # shellcheck disable=SC2086  # ^ name list
    key="$({ sha "$W/dump/dump.jsonl" "$W/dump/base.jsonl"; rr_json configs $CONFIGS; rr_json models $CONFIGS
        src_cat src/llm src/search/rerank_eval.rs src/search/hybrid.rs Cargo.lock scripts/beir-eval.py scripts/kaggle/run.py; } \
        | sha | cut -c1-24)"
    echo "$W/llm-cache/rr/$key/${1/\//-of-}"
}

# run_py <mode> [<i/N>]: scripts/kaggle/run.py on W/input. Results in W/working, state in W/root.
run_py() {
    rm -rf "$W/working"
    local configs=""
    case "$1" in
        dump) configs="$(rr_json configs)" ;;
        rerank)
            [[ -n "${CONFIGS:-}" ]] || die "mode=rerank needs configs (names in scripts/rerank-configs.json)"
            # shellcheck disable=SC2086  # ^ CONFIGS is a space-separated list of names
            configs="$(rr_json configs $CONFIGS)" ;;
    esac
    IRK_INPUT="$W/input" IRK_WORKING="$W/working" IRK_ROOT="$W/root" \
        IRK_DATASET="$DATASET" IRK_MODE="$1" IRK_GIT7="$GIT7" IRK_SIZE="$SIZE" IRK_PREPROCESSOR="$PRE" \
        IRK_SHARD="${2:-}" IRK_RERANK_CONFIGS="$configs" IRK_RERANK_DUMP_DIR="$W/dump" python3 "$RUN_PY"
    [[ "$1" != rerank ]] || return 0
    # ^ run.py passes --index-in only when the attached INDEX_KEY matches this build.
    grep -q "index: loaded eval-$LABEL" "$W/working/bench.log" \
        || die "run.py did not reuse ir-index-$DATASET: its INDEX_KEY does not match this build (key diff above). Rebuild it: eval-kaggle.yml dataset=$DATASET mode=vector"
}

keys() {
    local ds="$1" bundle="$2"
    W="$3"
    mkdir -p "$W"
    kaggle_auth || exit 2
    rm -rf "$W/key"
    kdl "ir-index-$ds" "$W/key" INDEX_KEY || true
    [[ -f "$W/key/INDEX_KEY" ]] || die "no INDEX_KEY in Kaggle dataset ir-index-$ds: build it with eval-kaggle.yml dataset=$ds mode=vector"
    local models f
    if [[ "${MODE:-}" == rerank ]]; then
        # ^ Nothing to restore: rerank shards fetch their GGUFs (fetch_models).
        emit MODELS_KEY eval-models-none
    else
        models="$(for f in embedding_file reranker_file expander_file; do bundle_field "$bundle" "$f"; done | sha)"
        emit MODELS_KEY "eval-models-${models:0:16}"
    fi
    emit DATA_KEY "eval-data-$ds-$(sha "$W/key/INDEX_KEY" | cut -c1-16)"
}

seed() {
    local ds="$1" bundle="$2" in fresh=0 label
    W="$3"
    in="$W/input"
    DATASET="$ds"
    dataset_spec "$ds" || exit 1
    kaggle_auth || exit 2
    mkdir -p "$in"
    rm -rf "$in/bundle"
    cp -R "$bundle" "$in/bundle"
    [[ -f "$in/models/$(bundle_field "$bundle" embedding_file)" ]] || kdl ir-models "$in/models"
    if [[ ! -f "$in/data/index/INDEX_KEY" ]]; then
        fresh=1
        rm -rf "$in/data"
        kdl "ir-corpus-$ds" "$in/data/corpus"
        kdl "ir-index-$ds" "$in/data/index"
        [[ -z "$PRE" ]] || kdl "ir-lindera-$PRE" "$in/data/lindera"
    fi
    # ^ The pool is the one the index was built on: run.py writes dataset=<bench.sh label>.
    label="$(sed -n 's/^dataset=//p' "$in/data/index/INDEX_KEY")"
    [[ "$label" == "$LABEL" ]] || die "ir-index-$ds was built on dataset=$label, not $LABEL: rebuild it with eval-kaggle.yml dataset=$ds mode=vector"
    local gguf
    gguf="$in/models/$(bundle_field "$bundle" reranker_file)"
    [[ -f "$gguf" ]] || die "no reranker $gguf in ir-models"
    emit GIT7 "$(bundle_field "$bundle" git7)"
    emit RERANK_KEY "$(sha "$gguf" "$REPO_ROOT/src/llm/scoring.rs" "$REPO_ROOT/src/llm/reranker.rs" | sha)"
    # ^ actions/cache key prefix of the llm_cache: the shard and merge jobs save under it.
    emit LLM_CACHE "llm-cache-$LABEL-$RERANK_KEY"
    # ^ What a dump depends on: the index, the reranker, and the code and script that produce the
    #   windows (src/search/rerank_eval.rs only reads dumps).
    emit DUMP_KEY "rr-dump-$LABEL-$({ cat "$in/data/index/INDEX_KEY"; echo "$RERANK_KEY"
        src_cat src/search src/db src/config src/llm/embedding.rs src/llm/mod.rs src/daemon.rs src/main.rs \
            scripts/bench.sh ':!src/search/rerank_eval.rs'; } | sha | cut -c1-24)"
    _log "seed: $LABEL at $GIT7; bm25 run as the INDEX_KEY gate"
    run_py bm25
    if [[ "$fresh" == 1 && -n "$SIZE" ]]; then
        # ^ Cache the pool, not the full corpus (MIRACL is GBs). bench.sh --size on a pool of at
        #   most that size samples every doc again (beir-eval.py sample), so the label holds.
        cp "$W/root/repo/test-data/$LABEL/corpus.jsonl" "$in/data/corpus/corpus.jsonl"
    fi
    if [[ "${MODE:-}" == rerank ]]; then
        # ^ The gate needed the bundle; the saved eval-models-none stays a marker.
        rm -rf "$in/models"
        mkdir -p "$in/models"
        echo "rerank shards fetch their models: scripts/eval-matrix.sh fetch_models" > "$in/models/NONE"
    fi
}

shard() {
    local s="$1" mode="$2" bundle="$3"
    W="$4"
    need_env
    rm -rf "$W/input/bundle"
    cp -R "$bundle" "$W/input/bundle"
    if [[ "$mode" != rerank ]]; then
        # ^ Always: also drops the rows of an index saved without this key. rerank never reads it.
        rerank_cache "$W/input/data/index/eval-$LABEL.sqlite" "$W/llm-cache/llm_cache.sqlite"
        run_py "$mode" "$s"
        return
    fi
    [[ -f "$W/dump/dump.jsonl" ]] || die "no dump in $W/dump: run mode=dump for $DATASET at this DUMP_KEY first"
    local saved part="$W/root/repo/logs/results/$LABEL/$GIT7.shard-${s/\//-of-}.json.partial"
    saved="$(rr_saved "$s")"
    if [[ -f "$saved/DONE" ]]; then
        _log "shard $s: reusing the partials of an earlier run ($saved)"
        mkdir -p "$part"
        cp "$saved"/*.jsonl "$part/"
        return
    fi
    fetch_models "$W/input/models"
    run_py rerank "$s"
    mkdir -p "$saved"
    cp "$part"/*.jsonl "$saved/"
    touch "$saved/DONE"
}

collect() {
    W="$1"
    local out="$2" state
    need_env
    mkdir -p "$out/$LABEL/qrels" "$W/llm-cache"
    # ^ beir-eval.py names the result's dataset after this directory.
    state="$W/root/repo/.bench-state/bench/xdg/ir/collections/eval-$LABEL.sqlite"
    [[ ! -f "$state" ]] || rerank_cache "$W/llm-cache/llm_cache.sqlite" "$state"
    # ^ Also without a bench.sh run (mode=rerank): the merged cache keeps the restored rows.
    [[ ! -f "$W/llm-cache/llm_cache.sqlite" ]] || cp "$W/llm-cache/llm_cache.sqlite" "$out/"
    [[ ! -d "$W/llm-cache/rr" ]] || cp -R "$W/llm-cache/rr" "$out/"
    [[ ! -d "$W/root/repo/logs/results/$LABEL" ]] || cp -R "$W/root/repo/logs/results/$LABEL" "$out/results"
    cp "$W/input/data/corpus/queries.jsonl" "$out/$LABEL/"
    cp "$W/input/data/corpus/test.tsv" "$out/$LABEL/qrels/"
    local f
    for f in bench.log daemon.log env.txt rerank.log dump.jsonl; do [[ ! -f "$W/working/$f" ]] || cp "$W/working/$f" "$out/"; done
}

merge() {
    local mode="$1" out="$3" s p data=""
    W="$2"
    shift 3
    need_env
    mkdir -p "$W/llm-cache" "$out"
    local caches=() parts=()
    for s in "$@"; do
        [[ ! -f "$s/llm_cache.sqlite" ]] || caches+=("$s/llm_cache.sqlite")
        [[ -n "$data" || ! -f "$s/$LABEL/queries.jsonl" ]] || data="$s/$LABEL"
        for p in "$s/results/$GIT7".shard-*.json.partial; do [[ ! -d "$p" ]] || parts+=("$p"); done
    done
    # ^ First, so the workflow saves the union even when a shard is missing.
    rerank_cache "$W/llm-cache/llm_cache.sqlite" ${caches[@]+"${caches[@]}"}
    cp "$W/llm-cache/llm_cache.sqlite" "$out/"
    for s in "$@"; do [[ ! -d "$s/rr" ]] || { mkdir -p "$W/llm-cache/rr"; cp -R "$s/rr/." "$W/llm-cache/rr/"; }; done
    [[ -n "$data" && ${#parts[@]} -gt 0 ]] || die "no shard results in: $*"
    local result="$out/logs/results/$LABEL/$GIT7.json"
    python3 "$BEIR" merge --data "$data" --output "$result" "${parts[@]}"
    if [[ "$mode" == dump ]]; then
        # ^ W/dump: the windows and the run's own rankings (base), for the rerank runs.
        rm -rf "$W/dump"
        mkdir -p "$W/dump"
        for s in "$@"; do [[ ! -f "$s/dump.jsonl" ]] || cat "$s/dump.jsonl" >> "$W/dump/dump.jsonl"; done
        cp "$result.partial/hybrid.jsonl" "$W/dump/base.jsonl"
        gzip -c "$W/dump/dump.jsonl" > "$out/dump.jsonl.gz"
        cp "$W/dump/base.jsonl" "$out/"
    fi
    if [[ "$mode" == dump || "$mode" == rerank ]]; then
        mode="$(cd "$result.partial" && ls -- *.jsonl | sed 's/\.jsonl$//' | paste -sd, -)"
    fi
    # ^ Every query has a record, so run only computes the metrics (it never calls ir).
    python3 "$BEIR" run --data "$data" --collection "eval-$LABEL" --mode "$mode" --output "$result"
}

cmd="${1:-}"
shift || true
case "$cmd" in
    keys) keys "$@" ;;
    seed) seed "$@" ;;
    shard) shard "$@" ;;
    collect) collect "$@" ;;
    merge) merge "$@" ;;
    *) sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
