#!/usr/bin/env bash
# Run IR benchmark against a BEIR dataset, optionally comparing two git revisions.
#
# Usage:
#   scripts/bench.sh <dataset> [baseline-ref] [--mode bm25|vector|hybrid|all] [--size N] [--seed N]
#
# Examples:
#   scripts/bench.sh fiqa
#   scripts/bench.sh fiqa v0.9.0
#   scripts/bench.sh miracl-ko
#   scripts/bench.sh fiqa v0.9.0 --mode bm25
#   scripts/bench.sh miracl-ko --size 10000
#
# Dataset names and their auto-detected settings:
#   fiqa, nfcorpus, scifact, arguana  — English BEIR datasets
#   miracl-ko                          — Korean MIRACL full corpus (download-miracl-ko.sh)
#   ko-miracl                          — Korean MIRACL dev split (download-ko-miracl.py)
#   miracl-zh                          — Chinese MIRACL full corpus (download-miracl-zh.sh)
#
# Results cached at logs/results/<dataset>/<git7>.json — reused on re-run.
# A per-mode score table is always printed from the cached JSON.
#
# On macOS, benchmark phases run under a safety watchdog by default. The watchdog
# keeps Metal enabled, but aborts the run if system free memory drops too low,
# swapouts begin, or `ir` sustains CPU-fallback-like usage for several checks.
# Override with:
#   IR_BENCH_GUARD=0              disable watchdog
#   IR_BENCH_MIN_FREE_PCT=12      abort sooner on low free memory
#   IR_BENCH_MAX_IR_CPU_PCT=600   abort sooner on CPU fallback
#   IR_BENCH_CPU_STRIKES=2        require fewer hot samples before aborting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "bench"
# Own the daemon lifecycle: on any exit (success, error, or Ctrl-C) stop this
# bench's isolated daemon and reap any in-flight guarded child. Safe for other
# sessions — IR_CONFIG_DIR is the isolated dir, so it never touches a work daemon.
trap bench_cleanup EXIT INT TERM

_log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Parse args ───────────────────────────────────────────────────────────────

DATASET=""
BASELINE_REF=""
MODE="all"
SAMPLE_SIZE=""
SAMPLE_SEED="42"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --size) SAMPLE_SIZE="$2"; shift 2 ;;
        --seed) SAMPLE_SEED="$2"; shift 2 ;;
        --) shift; break ;;
        -*)  echo "unknown flag: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$DATASET" ]]; then
                DATASET="$1"
            elif [[ -z "$BASELINE_REF" ]]; then
                BASELINE_REF="$1"
            else
                echo "unexpected argument: $1" >&2; exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "usage: $0 <dataset> [baseline-ref] [--mode bm25|vector|hybrid|all] [--size N] [--seed N]" >&2
    exit 1
fi

# ── Dataset config ────────────────────────────────────────────────────────────

case "$DATASET" in
    fiqa)
        DATA_PATH="test-data/fiqa"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh fiqa"
        EMBED_FLAG="--embed"
        ;;
    nfcorpus)
        DATA_PATH="test-data/nfcorpus"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh nfcorpus"
        EMBED_FLAG="--embed"
        ;;
    scifact)
        DATA_PATH="test-data/scifact"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh scifact"
        EMBED_FLAG="--embed"
        ;;
    arguana)
        DATA_PATH="test-data/arguana"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh arguana"
        EMBED_FLAG="--embed"
        ;;
    miracl-ko)
        DATA_PATH="test-data/miracl-ko"
        PREPROCESSOR="ko"
        DOWNLOAD_CMD="scripts/download-miracl-ko.sh"
        EMBED_FLAG="--embed"
        ;;
    ko-miracl)
        DATA_PATH="test-data/ko-miracl"
        PREPROCESSOR="ko"
        DOWNLOAD_CMD="uv run scripts/download-ko-miracl.py"
        EMBED_FLAG="--embed"
        ;;
    miracl-zh)
        DATA_PATH="test-data/miracl-zh"
        PREPROCESSOR="zh"
        DOWNLOAD_CMD="scripts/download-miracl-zh.sh"
        EMBED_FLAG="--embed"
        ;;
    *)
        echo "unknown dataset '$DATASET'. See script header for supported datasets." >&2
        exit 1
        ;;
esac

if [[ -n "$SAMPLE_SIZE" && ! "$SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid --size '$SAMPLE_SIZE' (expected positive integer)" >&2
    exit 1
fi

if [[ ! "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid --seed '$SAMPLE_SEED' (expected non-negative integer)" >&2
    exit 1
fi

# BM25-only mode: skip embedding
if [[ "$MODE" == "bm25" ]]; then
    EMBED_FLAG=""
fi

# Benchmarks should use the dedicated expander + reranker path unless explicitly
# overridden by the user. Auto-detecting a local Qwen combined GGUF makes the
# harness depend on ambient machine state instead of the intended benchmark path.
if [[ "$MODE" != "bm25" ]]; then
    unset IR_COMBINED_MODEL IR_QWEN_MODEL
    export IR_EXPANDER_MODEL="${IR_EXPANDER_MODEL:-tobil/qmd-query-expansion-1.7B}"
    export IR_RERANKER_MODEL="${IR_RERANKER_MODEL:-ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF}"
fi

# ── Download dataset if missing ───────────────────────────────────────────────

if [[ ! -f "$DATA_PATH/corpus.jsonl" ]]; then
    _log "Dataset '$DATASET' not found. Downloading..."
    eval "$DOWNLOAD_CMD"
fi

DATASET_LABEL="$DATASET"
if [[ -n "$SAMPLE_SIZE" ]]; then
    DATASET_LABEL="${DATASET}-s${SAMPLE_SIZE}-p${SAMPLE_SEED}"
    SAMPLE_PATH="test-data/${DATASET_LABEL}"
    if [[ ! -f "$SAMPLE_PATH/corpus.jsonl" ]]; then
        _log "Sampling '$DATASET' -> '$DATASET_LABEL'..."
        python3 scripts/beir-eval.py sample \
            --data "$DATA_PATH" \
            --size "$SAMPLE_SIZE" \
            --seed "$SAMPLE_SEED" \
            --output "$SAMPLE_PATH"
    fi
    DATA_PATH="$SAMPLE_PATH"
fi

# ── Build ir binary ───────────────────────────────────────────────────────────

_log "Building ir (HEAD)..."
cargo build --release --bin ir 2>&1
HEAD_BIN="$REPO_ROOT/target/release/ir"
HEAD_HASH=$(git rev-parse --short=7 HEAD)
_log "HEAD: $HEAD_HASH"

# ── Build baseline if requested ───────────────────────────────────────────────

BASELINE_BIN=""
BASELINE_HASH=""

if [[ -n "$BASELINE_REF" ]]; then
    BASELINE_HASH=$(git rev-parse --short=7 "$BASELINE_REF" 2>/dev/null) || {
        echo "ERROR: git ref '$BASELINE_REF' not found" >&2; exit 1
    }

    if [[ "$BASELINE_HASH" == "$HEAD_HASH" ]]; then
        _log "Baseline $BASELINE_REF ($BASELINE_HASH) matches HEAD — reusing binary"
        BASELINE_BIN="$HEAD_BIN"
    else
        WORKTREE="$REPO_ROOT/.bench-state/worktrees/$BASELINE_HASH"
        if [[ ! -d "$WORKTREE" ]]; then
            mkdir -p "$(dirname "$WORKTREE")"
            _log "Creating worktree for $BASELINE_REF ($BASELINE_HASH)..."
            git worktree add --quiet "$WORKTREE" "$BASELINE_REF"
        fi
        _log "Building ir ($BASELINE_REF)..."
        cargo build --release --bin ir --manifest-path "$WORKTREE/Cargo.toml" 2>&1
        BASELINE_BIN="$WORKTREE/target/release/ir"
        _log "baseline: $BASELINE_HASH"
    fi
fi

# ── Run one version ───────────────────────────────────────────────────────────

LOG_DIR="$REPO_ROOT/logs/results/$DATASET_LABEL"
mkdir -p "$LOG_DIR"

result_satisfies_mode() {
    local result_file="$1"
    local mode="$2"
    [[ -f "$result_file" ]] || return 1

    python3 - "$result_file" "$mode" <<'EOF'
import json
import sys

result_file, requested_mode = sys.argv[1:]
data = json.load(open(result_file))
modes = {r.get("mode") for r in data.get("results", [])}

if requested_mode == "all":
    ok = {"bm25", "vector", "hybrid"}.issubset(modes)
else:
    ok = requested_mode in modes

raise SystemExit(0 if ok else 1)
EOF
}

collection_ready_for_mode() {
    local collection="$1"
    local mode="$2"
    local db_path="$XDG_CONFIG_HOME/ir/collections/${collection}.sqlite"
    [[ -f "$db_path" ]] || return 1

    python3 - "$db_path" "$mode" <<'EOF'
import sqlite3
import sys

db_path, requested_mode = sys.argv[1:]
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    active = cur.execute(
        "SELECT COUNT(*) FROM documents WHERE active = 1"
    ).fetchone()[0]
    if active <= 0:
        raise SystemExit(1)
    if requested_mode == "bm25":
        raise SystemExit(0)
    pending = cur.execute(
        """
        SELECT COUNT(*)
        FROM documents d
        WHERE d.active = 1
          AND NOT EXISTS (
              SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash
          )
        """
    ).fetchone()[0]
    raise SystemExit(0 if pending == 0 else 1)
except sqlite3.Error:
    raise SystemExit(1)
EOF
}

mark_prepared() {
    local result_base="$1"
    local mode="$2"
    if [[ "$mode" == "bm25" ]]; then
        touch "${result_base}.prepared-bm25"
    else
        touch "${result_base}.prepared-bm25" "${result_base}.prepared-all"
    fi
}

prepared_marker_for_mode() {
    local result_base="$1"
    local mode="$2"
    if [[ "$mode" == "bm25" ]]; then
        printf "%s.prepared-bm25" "$result_base"
    else
        printf "%s.prepared-all" "$result_base"
    fi
}

run_version() {
    local label="$1"
    local git7="$2"
    local ir_bin="$3"
    local collection="eval-${DATASET_LABEL}-${git7}"
    local result_file="$LOG_DIR/${git7}.json"
    local result_base="${result_file%.json}"
    local prep_marker
    prep_marker=$(prepared_marker_for_mode "$result_base" "$MODE")

    if result_satisfies_mode "$result_file" "$MODE"; then
        _log "[$label $git7] cached ($result_file)"
        return 0
    fi

    if [[ -f "$prep_marker" ]]; then
        _log "[$label $git7] resume: prepared marker found ($prep_marker)"
    elif collection_ready_for_mode "$collection" "$MODE"; then
        _log "[$label $git7] resume: existing collection is ready — skipping prepare"
        mark_prepared "$result_base" "$MODE"
    else
        _log "[$label $git7] preparing collection..."
        prep_args=(
            prepare
            --ir-bin "$ir_bin"
            --data "$DATA_PATH"
            --collection "$collection"
        )
        [[ -n "$PREPROCESSOR" ]] && prep_args+=(--preprocessor "$PREPROCESSOR")
        [[ -n "$EMBED_FLAG" ]] && prep_args+=($EMBED_FLAG)
        bench_run_guarded "prepare $collection" "$ir_bin" python3 scripts/beir-eval.py "${prep_args[@]}"
        mark_prepared "$result_base" "$MODE"
    fi

    if [[ "$MODE" != "bm25" ]]; then
        _log "[$label $git7] restarting benchmark daemon (tier-2=dedicated)"
        "$ir_bin" daemon stop || true
    fi

    _log "[$label $git7] running queries (mode=$MODE)..."
    bench_run_guarded "query $collection ($MODE)" "$ir_bin" python3 scripts/beir-eval.py run \
        --ir-bin "$ir_bin" \
        --data "$DATA_PATH" \
        --collection "$collection" \
        --mode "$MODE" \
        --at-k "10,20,100" \
        --output "$result_file"

    _log "[$label $git7] done -> $result_file"
}

run_version "HEAD" "$HEAD_HASH" "$HEAD_BIN"
[[ -n "$BASELINE_REF" ]] && run_version "base" "$BASELINE_HASH" "$BASELINE_BIN"

# ── Print score table ─────────────────────────────────────────────────────────

print_table_rows() {
    local label="$1"
    local git7="$2"
    local result_file="$LOG_DIR/${git7}.json"

    if [[ ! -f "$result_file" ]]; then
        printf "  %-22s  %-7s  %-7s  %8s  %8s  %8s  %8s  %8s  %9s  %8s\n" \
               "$label" "$git7" "-" "MISSING" "-" "-" "-" "-" "-" "-"
        return
    fi

    python3 - "$result_file" "$label" "$git7" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
label = sys.argv[2]
git7 = sys.argv[3]

results = {r["mode"]: r for r in data["results"]}

def fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

for mode in ("bm25", "vector", "hybrid"):
    result = results.get(mode)
    if not result:
        continue
    metrics = result.get("metrics", {})
    timing = result.get("timing", {})
    n10 = metrics.get("ndcg_10", "?")
    n20 = metrics.get("ndcg_20", "?")
    r10 = metrics.get("recall_10", "?")
    r20 = metrics.get("recall_20", "?")
    r100 = metrics.get("recall_100", "?")
    r1k = metrics.get("recall_1000", "-")
    med = timing.get("median_ms", "-")
    print(
        f"  {label:<22}  {git7:<7}  {mode:<7}  "
        f"{fmt(n10):>8}  {fmt(n20):>8}  {fmt(r10):>8}  {fmt(r20):>8}  "
        f"{fmt(r100):>8}  {fmt(r1k):>9}  {fmt(med, 1) if isinstance(med, float) else str(med):>8}"
    )
EOF
}

echo ""
echo "══════════════════════════════════════════════════════════════════════════════════════"
echo "  Benchmark — $DATASET_LABEL  (mode=$MODE)"
echo "══════════════════════════════════════════════════════════════════════════════════════"
printf "  %-22s  %-7s  %-7s  %8s  %8s  %8s  %8s  %8s  %9s  %8s\n" \
       "run" "git" "mode" "nDCG@10" "nDCG@20" "R@10" "R@20" "R@100" "R@1000" "med ms"
printf "  %-22s  %-7s  %-7s  %8s  %8s  %8s  %8s  %8s  %9s  %8s\n" \
       "----------------------" "-------" "-------" "--------" "--------" "--------" "--------" "--------" "---------" "--------"

print_table_rows "HEAD" "$HEAD_HASH"
[[ -n "$BASELINE_REF" ]] && print_table_rows "$BASELINE_REF" "$BASELINE_HASH"

echo "══════════════════════════════════════════════════════════════════════════════════════"

# ── Cleanup worktree prompt ───────────────────────────────────────────────────

if [[ -n "$BASELINE_REF" && -n "$BASELINE_HASH" && "$BASELINE_HASH" != "$HEAD_HASH" ]]; then
    WORKTREE="/tmp/ir-bench-$BASELINE_HASH"
    if [[ -d "$WORKTREE" ]]; then
        echo ""
        read -rp "Remove worktree $WORKTREE? [y/N] " ans
        if [[ "$ans" =~ ^[Yy] ]]; then
            git worktree remove --force "$WORKTREE"
            echo "Removed."
        fi
    fi
fi
