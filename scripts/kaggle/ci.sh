#!/usr/bin/env bash
# Kaggle side of .github/workflows/eval-kaggle.yml. See research/kaggle-eval.md.
#
# Usage: scripts/kaggle/ci.sh <dataset> <mode> <accelerator> <bundle-dir> <out-dir>
#
# Auth comes from env only, never a file: KAGGLE_USERNAME plus KAGGLE_API_TOKEN
# (new token) or KAGGLE_KEY (legacy key). Needs the pinned `kaggle` CLI on PATH
# (scripts/kaggle/requirements.txt).
#
# Datasets (private, under $KAGGLE_USERNAME), each uploaded once:
#   ir-corpus-<dataset>  corpus.jsonl queries.jsonl test.tsv DATASET
#   ir-models            embedding + reranker GGUF (HF revisions pinned below)
#   ir-bench-bin         bundle.sh output; new version only when git7 changes
#   ir-index-<dataset>   built index; new version only when the kernel rebuilt it

set -euo pipefail

DATASET="${1:?dataset}"
MODE="${2:?mode}"
ACCEL="${3:?accelerator}"
BUNDLE="${4:?bundle dir}"
OUT="${5:?out dir}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT_MIN="${IRK_WAIT_MIN:-320}"

# ^ kaggle reads every KAGGLE_* env var as config; an empty secret would still
#   count as a credential, so drop empty ones.
for v in KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME; do
    [[ -n "${!v:-}" ]] || unset "$v"
done
[[ -n "${KAGGLE_USERNAME:-}" ]] || { echo "KAGGLE_USERNAME is not set" >&2; exit 2; }
[[ -n "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ]] || { echo "set KAGGLE_API_TOKEN or KAGGLE_KEY" >&2; exit 2; }
U="$KAGGLE_USERNAME"
git7_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["git7"])' "$1"; }
GIT7="$(git7_of "$BUNDLE/bundle.json")"
WORK="$(mktemp -d "${RUNNER_TEMP:-$REPO_ROOT/.ks}/irk.XXXXXX")"

_log() { echo "[$(date +%H:%M:%S)] $*"; }

# ^ kaggle 2.2.5 prints API errors from `datasets create|version` and `kernels push` but
#   exits 0 (kaggle_api_extended.py *_cli: "Dataset creation error:", "Kernel push error:",
#   "not valid dataset sources"). kcheck keeps the output in KOUT and fails on that text.
KOUT=""
kcheck() {
    KOUT="$("$@" 2>&1)" || { echo "$KOUT" >&2; return 1; }
    echo "$KOUT"
    if grep -Eqi 'error|not valid' <<<"$KOUT"; then
        echo "kaggle reported an error (exit code was 0): $*" >&2
        return 1
    fi
}

# dstate <slug>: "<status> <current_version_number>"; fails when the dataset does not exist.
dstate() {
    kaggle datasets status "$U/$1" --format json 2>/dev/null \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["status"], d["current_version_number"])'
}
exists() { dstate "$1" >/dev/null 2>&1; }

# wait_ready <slug> <min-version>: ready and at version >= min. A new version can leave the
# old one "ready" for a while, and a kernel would then mount the old files.
wait_ready() {
    local s=""
    for _ in $(seq 1 60); do
        s="$(dstate "$1" 2>/dev/null || true)"
        [[ "$s" == ready\ * && "${s##* }" -ge "$2" ]] && return 0
        sleep 20
    done
    echo "dataset $U/$1 not ready at version $2 after 20 min (state: $s)" >&2
    return 1
}

# publish <slug> <dir> <message>: create the dataset, or add a version.
publish() {
    local slug="$1" dir="$2" msg="$3"
    printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$slug" "$U" "$slug" \
        > "$dir/dataset-metadata.json"
    local want=1 state
    if state="$(dstate "$slug" 2>/dev/null)"; then
        want=$(( ${state##* } + 1 ))
        kcheck kaggle datasets version -p "$dir" -m "$msg" -q
    else
        kcheck kaggle datasets create -p "$dir" -q
    fi
    wait_ready "$slug" "$want"
}

ensure_corpus() {
    local slug="ir-corpus-$DATASET"
    exists "$slug" && return 0
    _log "seeding $slug"
    local src
    case "$DATASET" in
        fiqa|nfcorpus|scifact|arguana) "$REPO_ROOT/scripts/download-beir.sh" "$DATASET" ;;
        *) echo "no download for dataset $DATASET" >&2; return 1 ;;
    esac
    src="$REPO_ROOT/test-data/$DATASET"
    local dir="$WORK/$slug"
    mkdir -p "$dir"
    cp "$src/corpus.jsonl" "$src/queries.jsonl" "$dir/"
    cp "$src/qrels/test.tsv" "$dir/test.tsv"
    echo "$DATASET" > "$dir/DATASET"
    publish "$slug" "$dir" "seed $DATASET"
}

ensure_models() {
    exists ir-models && return 0
    _log "seeding ir-models"
    local dir="$WORK/ir-models"
    mkdir -p "$dir"
    # ^ Same HF revisions as the owner's local cache, so local and Kaggle runs embed alike.
    curl -fsSL -o "$dir/embeddinggemma-300M-Q8_0.gguf" \
        "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf"
    curl -fsSL -o "$dir/qwen3-reranker-0.6b-q8_0.gguf" \
        "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/a02f48bb4f057028298c21fa033da2b30d7742d5/qwen3-reranker-0.6b-q8_0.gguf"
    # ^ Off by the retrieval profile, but bench.sh names it and ir resolves every
    #   IR_*_MODEL before the daemon starts (llm/download.rs prepare_model_envs).
    curl -fsSL -o "$dir/qmd-query-expansion-1.7B-q4_k_m.gguf" \
        "https://huggingface.co/tobil/qmd-query-expansion-1.7B/resolve/c8ff0361379652a4b44ea9ed7595ddc6728905db/qmd-query-expansion-1.7B-Q4_K_M.gguf"
    publish ir-models "$dir" "seed models"
}

ensure_bundle() {
    local have=""
    if exists ir-bench-bin; then
        kaggle datasets download "$U/ir-bench-bin" -f bundle.json -p "$WORK/have" --unzip -q -o >/dev/null 2>&1 || true
        [[ -f "$WORK/have/bundle.json" ]] && \
            have="$(git7_of "$WORK/have/bundle.json")"
    fi
    if [[ "$have" == "$GIT7" ]]; then
        _log "ir-bench-bin already at $GIT7"
    else
        _log "uploading ir-bench-bin $GIT7 (was: ${have:-none})"
        publish ir-bench-bin "$BUNDLE" "$GIT7"
    fi
}

run_kernel() {
    local slug="ir-bench-$DATASET" kdir="$WORK/kernel" sources ref gpu=true
    local push=(kaggle kernels push -p "$kdir")
    # ^ bm25 loads no model: no GPU quota, no wait for a GPU slot (run.py stubs libcuda.so.1
    #   when the session has none).
    if [[ "$MODE" == bm25 ]]; then gpu=false; else push+=(--accelerator "$ACCEL"); fi
    sources="\"$U/ir-bench-bin\", \"$U/ir-models\", \"$U/ir-corpus-$DATASET\""
    exists "ir-index-$DATASET" && sources+=", \"$U/ir-index-$DATASET\""
    mkdir -p "$kdir"
    printf 'import os; os.environ.update({"IRK_DATASET": "%s", "IRK_MODE": "%s", "IRK_GIT7": "%s"})\n' \
        "$DATASET" "$MODE" "$GIT7" \
        > "$kdir/run.py"
    cat "$REPO_ROOT/scripts/kaggle/run.py" >> "$kdir/run.py"
    # ^ A "script" kernel got a CPU session although the server stored enable_gpu=true and
    #   machine_shape=NvidiaTeslaT4 (runs 43115533477, 43117637669, 43118549022; 30 h quota
    #   unused). Every working GPU push reported in Kaggle/kaggle-cli#490 is a notebook, so
    #   push run.py as a one-cell notebook. run.py raises on failure: a cell must fail the run.
    python3 - "$kdir/run.py" "$kdir/run.ipynb" <<'PY'
import json, sys
src = open(sys.argv[1]).read()
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[2], "w"))
PY
    cat > "$kdir/kernel-metadata.json" <<EOF
{
  "id": "$U/$slug",
  "title": "$slug",
  "code_file": "run.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": $gpu,
  "enable_internet": false,
  "dataset_sources": [$sources],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF
    _log "pushing $U/$slug (gpu=$gpu) with sources: $sources"
    # ^ Run 43115533477: the session had no GPU (no nvidia-smi, libcuda.so.1 missing). Print the
    #   weekly GPU quota and the settings the server stored, so a repeat shows the cause.
    kaggle quota 2>&1 || true
    kcheck "${push[@]}"
    mkdir -p "$WORK/pulled"
    kaggle kernels pull "$U/$slug" -p "$WORK/pulled" -m >/dev/null 2>&1 || true
    grep -E '"(enable_gpu|machine_shape|enable_internet|is_private)"' "$WORK/pulled/kernel-metadata.json" 2>/dev/null \
        | sed 's/^ */server: /' || true
    # ^ Status of the unversioned slug is the latest *finished* session until the new one
    #   starts, so a rerun could read the previous run's "complete". Pin the pushed version.
    ref="$U/$slug"
    local ver
    ver="$(sed -n 's/.*Kernel version \([0-9][0-9]*\) successfully.*/\1/p' <<<"$KOUT" | head -1)"
    if [[ -n "$ver" ]]; then ref="$ref/$ver"; else _log "no version in push output; polling the latest session"; fi

    local status=""
    for _ in $(seq 1 "$WAIT_MIN"); do
        sleep 60
        status="$(kaggle kernels status "$ref" 2>&1 | tr '[:upper:]' '[:lower:]' || true)"
        echo "$status"
        if [[ "$ref" != "$U/$slug" && "$status" != *"has status"* ]]; then
            _log "status of $ref failed; polling the latest session instead"
            ref="$U/$slug"
        fi
        case "$status" in
            *complete*|*error*|*cancel*) break ;;
        esac
    done

    mkdir -p "$OUT"
    kaggle kernels output "$ref" -p "$OUT" -o -q || true
    if [[ "$status" != *complete* ]]; then
        [[ -f "$OUT/$slug.log" ]] && tail -n 80 "$OUT/$slug.log"
        echo "kernel did not complete: $status" >&2
        return 1
    fi
    # ^ Output layout is not documented; search instead of assuming subdirs survive.
    [[ -n "$(find "$OUT" -name "$GIT7.json" -print -quit)" ]] \
        || { echo "kernel output has no $GIT7.json" >&2; return 1; }
}

save_index() {
    local key
    key="$(find "$OUT" -name INDEX_KEY -print -quit)"
    [[ -n "$key" ]] || { _log "index reused; no upload"; return 0; }
    _log "kernel rebuilt the index; uploading ir-index-$DATASET"
    publish "ir-index-$DATASET" "$(dirname "$key")" "rebuilt at $GIT7"
}

ensure_corpus
ensure_models
ensure_bundle
run_kernel
save_index
