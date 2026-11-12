#!/usr/bin/env bash
# CUDA build of .github/workflows/eval-kaggle.yml (job "build"), as a script so a change
# can be tried from a branch that touches scripts/ only. See research/kaggle-eval.md.
#
# Usage (inside nvidia/cuda:12.6.3-devel-ubuntu22.04, as root):
#   scripts/kaggle/build.sh deps             apt packages, with a mirror fallback
#   scripts/kaggle/build.sh build <out-dir>  cargo build, then bundle.sh into <out-dir>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

deps() {
    # ^ Canonical's DNS outage on 2026-11-11 (~13:48-17:19Z; actions/runner-images#15296,
    #   status.canonical.com) left *.ubuntu.com unresolvable for hours (runs 43034924699,
    #   43065784542). If apt fails, retry once on a mirror outside ubuntu.com; apt still
    #   checks the Ubuntu archive signatures. apt-get update exits 0 on fetch errors, so
    #   the failure shows in install.
    apt_install() {
        apt-get update \
            && apt-get install -y --no-install-recommends git curl ca-certificates cmake build-essential pkg-config libclang-dev python3
    }
    apt_install || {
        sed -i -E 's#http://(archive|security)\.ubuntu\.com/ubuntu#http://mirrors.edge.kernel.org/ubuntu#' /etc/apt/sources.list
        apt_install
    }
}

build() {
    local out="${1:?usage: build.sh build <out-dir>}"
    # ^ T4 only; Kaggle retired P100 (kaggle CLI 2.2.5 help).
    export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-75}"
    export CMAKE_DISABLE_FIND_PACKAGE_OpenSSL=TRUE
    # ^ numkong 7.7.1 (usearch dep) enables its sapphireamx kernels on GCC 11, but they need
    #   target("avx512fp16") (GCC 12+). build.rs override; run 43019698386 failed on it.
    export NK_TARGET_SAPPHIREAMX=0
    # ^ ggml GGML_CUDA_NCCL defaults ON and finds the image's libnccl, but llama-cpp-sys-2
    #   0.1.145 does not link it (run 43028557699: undefined ncclAllReduce). Kaggle runs one
    #   GPU. build.rs passes CMAKE_* env to cmake but does not rerun when it changes: change
    #   the rust-cache key if a cached build predates a CMAKE_* edit.
    export CMAKE_DISABLE_FIND_PACKAGE_NCCL=TRUE
    # ^ Without a target-cpu, llama-cpp-sys-2 builds ggml-cpu for SSE2 only (GGML_NATIVE OFF; AVX
    #   flags come from CARGO_CFG_TARGET_FEATURE). Kaggle sessions without a GPU run every model
    #   on that backend: build for AVX2/FMA/F16C. run.py checks the CPU flags before it starts.
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64-v3"
    cd "$REPO_ROOT"
    cargo build --release --bin ir --no-default-features --features llama-cuda,llama-openmp
    # ^ A tarball checkout (no git in the image when checkout ran) has no .git: name the build
    #   from the commit the workflow runs on.
    if [[ -z "${IRK_GIT7:-}" && -n "${GITHUB_SHA:-}" ]]; then export IRK_GIT7="${GITHUB_SHA:0:7}"; fi
    scripts/kaggle/bundle.sh target/release/ir "$out"
}

case "${1:-}" in
    deps) deps ;;
    build) build "${2:-}" ;;
    *) echo "usage: build.sh deps | build <out-dir>" >&2; exit 2 ;;
esac
