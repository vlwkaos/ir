# Shared by scripts/kaggle/ci.sh and scripts/eval-matrix.sh (sourced). See research/kaggle-eval.md.

_log() { echo "[$(date +%H:%M:%S)] $*"; }
sha() { shasum -a 256 "$@" | cut -d' ' -f1; }
# bundle_field <bundle-dir> <field>: a field of bundle.json (scripts/kaggle/bundle.sh).
bundle_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1/bundle.json" "$2"; }

# dataset_spec <dataset>: the eval datasets of both workflows. Sets SIZE (bench.sh --size, seed 42;
# empty: full corpus), PRE (bench.sh's preprocessor alias), DICT (its lindera dictionary) and
# LABEL (bench.sh DATASET_LABEL). fiqa and MIRACL run on a 10000-doc pool: research/kaggle-eval.md, Pool.
dataset_spec() {
    SIZE="" PRE="" DICT=""
    case "$1" in
        scifact|nfcorpus) ;;
        fiqa) SIZE=10000 ;;
        miracl-ko) SIZE=10000 PRE=ko DICT=ko-dic ;;
        miracl-zh) SIZE=10000 PRE=zh DICT=jieba ;;
        *) echo "unknown dataset '$1'" >&2; return 1 ;;
    esac
    LABEL="$1${SIZE:+-s$SIZE-p42}"
}

# kaggle_auth: kaggle reads every KAGGLE_* env var as config, and an empty secret would still count
# as a credential, so drop empty ones. Auth comes from env only, never a file.
kaggle_auth() {
    local v
    for v in KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME; do [[ -n "${!v:-}" ]] || unset "$v"; done
    [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ]] \
        || { echo "set KAGGLE_USERNAME and KAGGLE_API_TOKEN or KAGGLE_KEY" >&2; return 2; }
}

# kdl <slug> <dir> [file]: download a dataset of $KAGGLE_USERNAME (or one file of it), unzipped.
kdl() {
    local args=(kaggle datasets download "$KAGGLE_USERNAME/$1" -p "$2" --unzip -q -o)
    [[ -z "${3:-}" ]] || args+=(-f "$3")
    mkdir -p "$2"
    _log "kaggle: $KAGGLE_USERNAME/$1${3:+ ($3)}"
    "${args[@]}"
}
