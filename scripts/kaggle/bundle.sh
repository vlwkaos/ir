#!/usr/bin/env bash
# Assemble the Kaggle bench bundle: ir binary + bench scripts + bundle.json.
# See research/kaggle-eval.md.
#
# Usage: [IRK_GIT7=<7 hex>] scripts/kaggle/bundle.sh <ir-binary> <out-dir>
# IRK_GIT7 names the build when the checkout has no .git (container without git).
#
# bundle.json "index_key_src" lists what a built index depends on, except the
# embedding GGUF bytes (run.py adds their sha256 on Kaggle). An index is reused
# only when the full key matches, so a change here forces a rebuild.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IR_BIN="${1:?usage: bundle.sh <ir-binary> <out-dir>}"
OUT="${2:?usage: bundle.sh <ir-binary> <out-dir>}"
[[ -x "$IR_BIN" ]] || { echo "not an executable: $IR_BIN" >&2; exit 1; }
cd "$REPO_ROOT"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
# ^ First `pub const NAME: &str = "file";` in src/llm/mod.rs (models::NAME).
model_file() { sed -n "s/^ *pub const $1: &str = \"\(.*\)\";/\1/p" src/llm/mod.rs | head -1; }

EMBEDDING_FILE="$(model_file EMBEDDING)"
RERANKER_FILE="$(model_file RERANKER)"
EXPANDER_FILE="$(model_file EXPANDER)"
[[ -n "$EMBEDDING_FILE" && -n "$RERANKER_FILE" && -n "$EXPANDER_FILE" ]] || { echo "model consts not found in src/llm/mod.rs" >&2; exit 1; }

# ^ Embedding model name, chunking, embedding input format, HNSW build, schema.
KEY_SRC="embedding_file=$EMBEDDING_FILE
chunker.rs=$(sha src/index/chunker.rs)
embed.rs=$(sha src/index/embed.rs)
embedding.rs=$(sha src/llm/embedding.rs)
ann.rs=$(sha src/db/ann.rs)
schema.rs=$(sha src/db/schema.rs)
"

mkdir -p "$OUT"
cp "$IR_BIN" "$OUT/ir"
cp scripts/bench.sh scripts/bench-env.sh scripts/beir-eval.py "$OUT/"
# ^ Kaggle's image may lack libgomp, which the llama-openmp build links dynamically. Ship the
#   build host's copy flat (Kaggle datasets skip directories); run.py adds it to LD_LIBRARY_PATH.
if command -v ldd >/dev/null 2>&1; then
    { ldd "$IR_BIN" || true; } | awk '$1 ~ /^libgomp\.so/ && $3 ~ /^\// {print $1, $3}' | while read -r name path; do
        cp -L "$path" "$OUT/$name"
    done
fi
GIT7="${IRK_GIT7:-$(git rev-parse --short=7 HEAD)}"
[[ "$GIT7" =~ ^[0-9a-f]{7}$ ]] || { echo "bad git7 '$GIT7'" >&2; exit 1; }
python3 - "$OUT/bundle.json" "$GIT7" "$EMBEDDING_FILE" "$RERANKER_FILE" "$EXPANDER_FILE" "$KEY_SRC" <<'PY'
import json, sys
out, git7, emb, rer, exp, key = sys.argv[1:]
json.dump({"git7": git7, "embedding_file": emb, "reranker_file": rer,
           "expander_file": exp, "index_key_src": key}, open(out, "w"), indent=2)
PY
echo "bundle $GIT7 -> $OUT"
