#!/usr/bin/env bash
# One-off research driver: rerank-window sweep on full-corpus fiqa.
# Isolates the window axis — all arms share ONE collection (eval-fiqa-signals,
# full 57.6k fiqa) with graph + ANN forced OFF, so ONLY rerank_window varies.
# Pipeline per arm = the 0.18 default (expander off, keep-window on) except window.
# Idempotent + resumable: skips arms whose iso-w<N>.json exists; in-progress arms
# resume from logs/results/fiqa/iso-w<N>.json.partial/hybrid.jsonl.
#
# Usage:  bash scripts/window-sweep.sh            # all arms 20 40 60 100
#         bash scripts/window-sweep.sh 20 40      # subset
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WINDOWS=("$@"); [[ ${#WINDOWS[@]} -eq 0 ]] && WINDOWS=(20 40 60 100)
BIN=target/release/ir
COL=eval-fiqa-signals
DATA=test-data/fiqa
OUTDIR=logs/results/fiqa

export IR_GRAPH_T0_EXPAND=0 IR_ANN=0 IR_RERANK_KEEP_WINDOW=1
export IR_RERANKER_MODEL="${IR_RERANKER_MODEL:-ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF}"
unset IR_COMBINED_MODEL IR_QWEN_MODEL 2>/dev/null || true

_log() { echo "[$(date +%H:%M:%S)] $*"; }

for W in "${WINDOWS[@]}"; do
    OUT="$OUTDIR/iso-w${W}.json"
    if [[ -f "$OUT" ]]; then
        _log "w${W}: cached ($OUT) — skip"
        continue
    fi
    done_q=$(wc -l < "$OUT.partial/hybrid.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
    _log "w${W}: starting (resume from ${done_q:-0}/648) — restarting daemon"
    export IR_RERANK_WINDOW_OVERRIDE="$W"
    "$BIN" daemon stop 2>/dev/null || true
    python3 scripts/beir-eval.py run \
        --ir-bin "$BIN" --data "$DATA" --collection "$COL" \
        --mode hybrid --at-k 10,20,100 --output "$OUT" 2>/dev/null | tail -1
    if [[ -f "$OUT" ]]; then
        _log "w${W}: COMPLETE -> $OUT"
    else
        _log "w${W}: incomplete (timed out) — re-run this script to resume"
        exit 0
    fi
done
_log "sweep complete for windows: ${WINDOWS[*]}"
