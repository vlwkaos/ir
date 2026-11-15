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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PY="$REPO_ROOT/scripts/kaggle/run.py"
BEIR="$REPO_ROOT/scripts/beir-eval.py"
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

# run_py <mode> [<i/N>]: scripts/kaggle/run.py on W/input. Results in W/working, state in W/root.
run_py() {
    rm -rf "$W/working"
    IRK_INPUT="$W/input" IRK_WORKING="$W/working" IRK_ROOT="$W/root" \
        IRK_DATASET="$DATASET" IRK_MODE="$1" IRK_GIT7="$GIT7" IRK_SIZE="$SIZE" IRK_PREPROCESSOR="$PRE" \
        IRK_SHARD="${2:-}" python3 "$RUN_PY"
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
    local models
    models="$(for f in embedding_file reranker_file expander_file; do bundle_field "$bundle" "$f"; done | sha)"
    emit MODELS_KEY "eval-models-${models:0:16}"
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
    [[ -n "$(ls -A "$in/models" 2>/dev/null)" ]] || kdl ir-models "$in/models"
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
    _log "seed: $LABEL at $GIT7; bm25 run as the INDEX_KEY gate"
    run_py bm25
    if [[ "$fresh" == 1 && -n "$SIZE" ]]; then
        # ^ Cache the pool, not the full corpus (MIRACL is GBs). bench.sh --size on a pool of at
        #   most that size samples every doc again (beir-eval.py sample), so the label holds.
        cp "$W/root/repo/test-data/$LABEL/corpus.jsonl" "$in/data/corpus/corpus.jsonl"
    fi
}

shard() {
    local s="$1" mode="$2" bundle="$3"
    W="$4"
    need_env
    rm -rf "$W/input/bundle"
    cp -R "$bundle" "$W/input/bundle"
    # ^ Always: also drops the rows of an index saved without this key.
    rerank_cache "$W/input/data/index/eval-$LABEL.sqlite" "$W/llm-cache/llm_cache.sqlite"
    run_py "$mode" "$s"
}

collect() {
    W="$1"
    local out="$2" state
    need_env
    mkdir -p "$out/$LABEL/qrels" "$W/llm-cache"
    # ^ beir-eval.py names the result's dataset after this directory.
    state="$W/root/repo/.bench-state/bench/xdg/ir/collections/eval-$LABEL.sqlite"
    if [[ -f "$state" ]]; then
        rerank_cache "$W/llm-cache/llm_cache.sqlite" "$state"
        cp "$W/llm-cache/llm_cache.sqlite" "$out/"
    fi
    [[ ! -d "$W/root/repo/logs/results/$LABEL" ]] || cp -R "$W/root/repo/logs/results/$LABEL" "$out/results"
    cp "$W/input/data/corpus/queries.jsonl" "$out/$LABEL/"
    cp "$W/input/data/corpus/test.tsv" "$out/$LABEL/qrels/"
    local f
    for f in bench.log daemon.log env.txt; do [[ ! -f "$W/working/$f" ]] || cp "$W/working/$f" "$out/"; done
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
    [[ -n "$data" && ${#parts[@]} -gt 0 ]] || die "no shard results in: $*"
    local result="$out/logs/results/$LABEL/$GIT7.json"
    python3 "$BEIR" merge --data "$data" --output "$result" "${parts[@]}"
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
