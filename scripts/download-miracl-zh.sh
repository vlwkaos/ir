#!/usr/bin/env bash
# download-miracl-zh.sh: MIRACL Chinese dev split in BEIR format
# Output: test-data/miracl-zh/{corpus.jsonl, queries.jsonl, qrels/test.tsv}
# Stdlib Python; see scripts/download-miracl.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 scripts/download-miracl.py zh test-data/miracl-zh
