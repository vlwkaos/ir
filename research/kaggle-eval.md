# Retrieval eval on Kaggle (GPU or CPU)

`scripts/bench.sh` quality runs (nDCG, recall) on Kaggle's free tier, started from
GitHub Actions (`.github/workflows/eval-kaggle.yml`, `workflow_dispatch`). `run.py` uses the T4
when the session has one and the CPU otherwise. Speed numbers from Kaggle are not comparable
with a local GPU: preship floors do not apply.

```mermaid
flowchart LR
  subgraph GH["GitHub Actions"]
    B["build.sh: nvidia/cuda 12.6 container<br/>llama-cuda, sm_75"] --> BU["bundle.sh<br/>ir + bench scripts + bundle.json"]
    BU --> CI["ci.sh"]
  end
  subgraph KD["Kaggle datasets (private, read-only in the kernel)"]
    C["ir-corpus-&lt;ds&gt;"]
    M["ir-models"]
    BIN["ir-bench-bin"]
    I["ir-index-&lt;ds&gt;"]
  end
  CI -->|"once"| C & M
  CI -->|"new git7 only"| BIN
  CI -->|"kernels push, internet off"| K["run.py: GPU if /dev/nvidia*, else CPU"]
  KD --> K
  K -->|"INDEX_KEY match"| Q["bench.sh --index-in: query only"]
  K -->|"no match"| P["bench.sh --index-out: prepare + embed, then query"]
  P -->|"kernels output, datasets version"| I
  Q & P -->|"kernels output"| A["artifact eval-&lt;ds&gt;:<br/>result/&lt;ds&gt;/&lt;git7&gt;.json, bench.log, env.txt, daemon.log"]
```

## Setup (once)

1. Kaggle account with phone verification (GPU needs it).
2. Repo secrets: `KAGGLE_USERNAME`, plus `KAGGLE_API_TOKEN` (new token) or `KAGGLE_KEY` (legacy key).
   The workflow reads them as env vars only; nothing writes a token file.
3. Run the workflow. The first run seeds the corpus and model datasets, then builds the index on Kaggle.

## Rules

| What | Rule |
|---|---|
| Index reuse | `run.py` compares `INDEX_KEY`: `bundle.json` `index_key_src` (embedding file name, sha256 of `chunker.rs`, `embed.rs`, `embedding.rs`, `ann.rs`, `schema.rs`) + dataset + sha256 of the embedding GGUF. Any change rebuilds. |
| bm25 runs | Never save an index (bm25 prepare skips embedding); pushed with `enable_gpu` false |
| Collection name | `eval-<ds>` with `--index-in`/`--index-out`, `eval-<ds>-<git7>` otherwise |
| Read-only inputs | Index files are copied to `/tmp/irk` (search opens the DB read-write); the corpus is symlinked |
| No download | Kernel runs with `enable_internet: false`; models come from `ir-models` by path |
| CUDA build env | `NK_TARGET_SAPPHIREAMX=0` (numkong AMX kernels need GCC 12+, run 43019698386); `CMAKE_DISABLE_FIND_PACKAGE_NCCL=TRUE` (ggml finds the image's NCCL but llama-cpp-sys-2 does not link it, run 43028557699). A CMAKE_* change does not rerun llama-cpp-sys-2's build.rs: change the rust-cache key |
| CLI exit codes | `kaggle` 2.2.5 exits 0 after "Dataset creation error", "Dataset version creation error", "Kernel push error" and "not valid dataset sources" (`*_cli` in `kaggle_api_extended.py`). `ci.sh` `kcheck` fails on that text |
| Dataset versions | `ci.sh` waits for `status=ready` and `current_version_number` above the old one (`datasets status --format json`); a kernel could mount the old files otherwise. `run.py` also checks `IRK_GIT7` against `bundle.json` |
| Kernel status | `ci.sh` pins the pushed version (`kernels status <owner>/<slug>/<N>`), else a rerun can read the previous run's `complete` |
| No GPU from the API | Kernels pushed by `kaggle kernels push` got a CPU session (4 vCPU Xeon 2.2 GHz, no `/dev/nvidia*`, no driver libraries), although the server stored `enable_gpu` true and `machine_shape` NvidiaTeslaT4 and the weekly quota was unused (probe run 43120474122). Cause not found; check the account in the Kaggle web UI. `ci.sh` prints `kaggle quota` and the stored settings |
| CPU fallback | No `/dev/nvidia*` and no `nvidia-smi -L`: `run.py` runs the same CUDA build on the CPU backend (`IR_FORCE_CPU_BACKEND=1`, `IR_GPU_LAYERS=0`). `ir` has `libcuda.so.1` in NEEDED and imports 11 `cu*` driver functions, so `run.py` builds a stub that defines them; ggml-cuda then finds no device. The log says `[irk] backend: CPU` (run 43147559287) |
| x86-64-v3 build | `build.sh` adds `-C target-cpu=x86-64-v3`: llama-cpp-sys-2 turns `GGML_NATIVE` off and takes AVX flags only from the Rust target features, so ggml-cpu would be SSE2-only. `run.py` stops when the CPU lacks avx2/fma/f16c/bmi2; `env.txt` counts `vfmadd` in `ir` (7545) |
| CPU timeouts | A tier-2 query on 4 vCPU can take minutes. On CPU `run.py` sets `IR_DAEMON_REPLY_TIMEOUT=1800` (else a `-q` search silently falls back to BM25 after 120 s) and `IR_BENCH_QUERY_TIMEOUT=1900`. `beir-eval.py` reports `failed_queries` and `timing.total_s` per mode |
| Session outlasts the job | The Kaggle job waits `IRK_WAIT_MIN` (320) min inside a 350 min job; a CPU session can run 12 h. Dispatch the same ref, dataset and mode again: `ci.sh` compares the first line of the latest session (dataset, mode, git7) and attaches while it runs, or collects its output when it is complete (no new push and no dataset upload; run 43226621572). To run the same commit again, push a new commit. Another header still running fails the job. A session that Kaggle stopped (`cancel_acknowledged`, 12 h) or any other session: input `fetch` |
| Kaggle image | Ubuntu 24.04, glibc 2.39, Python 3.13, `gcc` present; datasets mount at `/kaggle/input/datasets/<owner>/<slug>` (run 43115533477). The build container (Ubuntu 22.04, glibc 2.34 symbols) runs there |
| libgomp | `bundle.sh` ships the build host's `libgomp.so.*` flat in the bundle (Kaggle datasets skip directories); `run.py` puts it on `LD_LIBRARY_PATH` |
| Auth | `kaggle` 2.2.5 tries `KAGGLE_API_TOKEN` first, then `KAGGLE_USERNAME` + `KAGGLE_KEY` (`kaggle_api_extended.py` `authenticate`) |

## How to run

```bash
gh workflow run eval-kaggle.yml -f dataset=scifact -f mode=vector       # build, push, wait, artifact eval-scifact
gh workflow run eval-kaggle.yml -f fetch=ir-bench-scifact/8            # only download that session's output: artifact kaggle-output
```

The kernel slug is `ir-bench-<dataset>`; the version is in the job log (`Kernel version N successfully pushed`).

## CPU timings (scifact, 4 vCPU Xeon 2.2 GHz, 300 queries)

| Step | Wall time | nDCG@10 (local Mac) | Source |
|---|---|---|---|
| prepare + embed (5858 chunks) | 7036 s (1 h 57 min) | n/a | run 43147559287 |
| reuse the index dataset | under 1 min (`index: key match`, no `preparing collection`) | n/a | run 43168270510 (kernel 8) |
| bm25 | 72.8 s | 0.6912 (0.6912) | kernel 8 |
| vector | 73.4 s | 0.7706 (0.7717) | kernel 8; also run 43147559287 |
| hybrid | about 255 s per query; 167/300 after 11 h 49 min, stopped at 12 h | none | kernel 8, collected by run 43226621572 |

- Hybrid does not fit a CPU session: scifact needs about 21 h. Each tier-2 query reranks 100 documents with the 0.6B model.
- fiqa does not fit either: its embed alone is about 6.9× scifact (by words), about 13.5 h.
- On CPU, run `bm25` and `vector`. Hybrid needs the GPU, or the queries split across sessions (not built).

## Trying a change without the owner

The build job runs `scripts/kaggle/build.sh` and the Kaggle job runs `ci.sh` and `run.py`.
The `gh` token in wsx has no `workflow` scope: it cannot push `.github/workflows/`. A branch
off `main` that changes `scripts/` and `research/` only can be pushed, and runs with
`gh workflow run eval-kaggle.yml --ref <branch> -f dataset=scifact`. A workflow edit runs only after it lands on `main`.

## Local dry run (no Kaggle)

```bash
scripts/kaggle/bundle.sh target/release/ir <input>/bin
# <input>/corpus: corpus.jsonl queries.jsonl test.tsv DATASET; <input>/models: the two GGUF files
IRK_DATASET=scifact IRK_INPUT=<input> IRK_WORKING=<work> IRK_ROOT=<short-dir> python3 scripts/kaggle/run.py
```

Move `<work>/index` under `<input>` and run again: the log shows `index: key match` and no `preparing collection`.

## Not yet

- MIRACL ko/zh: the kernel needs a Linux lindera preprocessor (`config.yml` aliases under
  `HOME/.config/ir`), and its version must go into `INDEX_KEY`.

## Limits (sources checked 2026-11)

- GPU quota about 30 h/week; T4 x2 default, P100 retired (`kaggle` 2.2.5 `--accelerator` help).
- Private dataset quota 200 GB. Output (`/kaggle/working`) about 20 GB.
- GitHub job limit 6 h; `IRK_WAIT_MIN` (default 320) bounds the poll.
