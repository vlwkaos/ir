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

exists() { kaggle datasets status "$U/$1" >/dev/null 2>&1; }

wait_ready() {
    local s=""
    for _ in $(seq 1 60); do
        s="$(kaggle datasets status "$U/$1" 2>/dev/null || true)"
        [[ "$s" == *ready* ]] && return 0
        sleep 20
    done
    echo "dataset $U/$1 not ready after 20 min (status: $s)" >&2
    return 1
}

# publish <slug> <dir> <message>: create the dataset, or add a version.
publish() {
    local slug="$1" dir="$2" msg="$3"
    printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$slug" "$U" "$slug" \
        > "$dir/dataset-metadata.json"
    if exists "$slug"; then
        kaggle datasets version -p "$dir" -m "$msg" -q
    else
        kaggle datasets create -p "$dir" -q
    fi
    wait_ready "$slug"
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
        kaggle datasets download "$U/ir-bench-bin" -f bundle.json -p "$WORK/have" -q -o >/dev/null 2>&1 || true
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
    local slug="ir-bench-$DATASET" kdir="$WORK/kernel" sources gpu=true
    local push=(kaggle kernels push -p "$kdir")
    # ^ bm25 loads no model; do not spend GPU quota or wait for a GPU slot.
    if [[ "$MODE" == bm25 ]]; then gpu=false; else push+=(--accelerator "$ACCEL"); fi
    sources="\"$U/ir-bench-bin\", \"$U/ir-models\", \"$U/ir-corpus-$DATASET\""
    exists "ir-index-$DATASET" && sources+=", \"$U/ir-index-$DATASET\""
    mkdir -p "$kdir"
    printf 'import os; os.environ.update({"IRK_DATASET": "%s", "IRK_MODE": "%s"})\n' "$DATASET" "$MODE" \
        > "$kdir/run.py"
    cat "$REPO_ROOT/scripts/kaggle/run.py" >> "$kdir/run.py"
    cat > "$kdir/kernel-metadata.json" <<EOF
{
  "id": "$U/$slug",
  "title": "$slug",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
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
    "${push[@]}"

    local status=""
    for _ in $(seq 1 "$WAIT_MIN"); do
        sleep 60
        status="$(kaggle kernels status "$U/$slug" 2>&1 | tr '[:upper:]' '[:lower:]' || true)"
        echo "$status"
        case "$status" in
            *complete*|*error*|*cancel*) break ;;
        esac
    done

    mkdir -p "$OUT"
    kaggle kernels output "$U/$slug" -p "$OUT" -o -q || true
    [[ "$status" == *complete* ]] || { echo "kernel did not complete: $status" >&2; return 1; }
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
