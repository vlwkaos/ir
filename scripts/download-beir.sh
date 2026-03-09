#!/usr/bin/env bash
# Download BEIR datasets into test-data/
# Usage: scripts/download-beir.sh [dataset ...]
# Default: nfcorpus scifact fiqa arguana trec-covid

set -euo pipefail

BASE="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DEST="test-data"
DATASETS=("${@:-nfcorpus scifact fiqa arguana trec-covid}")

# Flatten the default into an array when called with no args
if [ "$#" -eq 0 ]; then
    DATASETS=(nfcorpus scifact fiqa arguana trec-covid)
fi

mkdir -p "$DEST"

for ds in "${DATASETS[@]}"; do
    dir="$DEST/$ds"
    if [ -d "$dir" ]; then
        echo "  $ds: already present, skipping"
        continue
    fi
    zip="$DEST/$ds.zip"
    echo "  $ds: downloading..."
    curl -fL --progress-bar -o "$zip" "$BASE/$ds.zip"
    echo "  $ds: extracting..."
    unzip -q "$zip" -d "$DEST"
    rm "$zip"
    echo "  $ds: done ($(find "$dir/corpus.jsonl" -printf "%s bytes\n" 2>/dev/null || echo "ok"))"
done

echo "all datasets ready in $DEST/"
