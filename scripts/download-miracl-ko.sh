#!/usr/bin/env bash
# download-miracl-ko.sh: MIRACL Korean dev split in BEIR format
# Output: test-data/miracl-ko/{corpus.jsonl, queries.jsonl, qrels/test.tsv}
# Stdlib Python; see scripts/download-miracl.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 scripts/download-miracl.py ko test-data/miracl-ko
