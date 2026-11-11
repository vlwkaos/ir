# Retrieval eval on a Kaggle GPU

`scripts/bench.sh` quality runs (nDCG, recall) on Kaggle's free T4, started from
GitHub Actions (`.github/workflows/eval-kaggle.yml`, `workflow_dispatch`). Speed
numbers from Kaggle are not comparable with a local GPU: preship floors do not apply.

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
  CI -->|"kernels push, T4, internet off"| K["run.py"]
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
| bm25 runs | Never save an index (bm25 prepare skips embedding); the kernel runs without a GPU |
| Collection name | `eval-<ds>` with `--index-in`/`--index-out`, `eval-<ds>-<git7>` otherwise |
| Read-only inputs | Index files are copied to `/tmp/irk` (search opens the DB read-write); the corpus is symlinked |
| No download | Kernel runs with `enable_internet: false`; models come from `ir-models` by path |
| CUDA build env | `NK_TARGET_SAPPHIREAMX=0` (numkong AMX kernels need GCC 12+, run 43019698386); `CMAKE_DISABLE_FIND_PACKAGE_NCCL=TRUE` (ggml finds the image's NCCL but llama-cpp-sys-2 does not link it, run 43028557699). A CMAKE_* change does not rerun llama-cpp-sys-2's build.rs: change the rust-cache key |
| CLI exit codes | `kaggle` 2.2.5 exits 0 after "Dataset creation error", "Dataset version creation error", "Kernel push error" and "not valid dataset sources" (`*_cli` in `kaggle_api_extended.py`). `ci.sh` `kcheck` fails on that text |
| Dataset versions | `ci.sh` waits for `status=ready` and `current_version_number` above the old one (`datasets status --format json`); a kernel could mount the old files otherwise. `run.py` also checks `IRK_GIT7` against `bundle.json` |
| Kernel status | `ci.sh` pins the pushed version (`kernels status <owner>/<slug>/<N>`), else a rerun can read the previous run's `complete` |
| No GPU from the API | Kernels pushed by `kaggle kernels push` got a CPU session (4 vCPU, no `/dev/nvidia*`, no driver libraries), although the server stored `enable_gpu` true and `machine_shape` NvidiaTeslaT4 and the weekly quota was unused. Same for script and notebook, internet on or off, `machine_shape` set or not (probe run 43120474122). Cause not found; check the account (phone verification) and that a GPU session works in the Kaggle web UI. `ci.sh` prints `kaggle quota` and the stored settings, `env.txt` lists the mounted GPU files |
| bm25 without a GPU | `ir` has `libcuda.so.1` in NEEDED and imports 11 `cu*` driver functions (ggml-cuda takes their address). `run.py` builds a stub that defines them when the session has no libcuda, so bm25 runs on a CPU kernel (run 43122638298). Other modes need a real GPU |
| Kaggle image | Ubuntu 24.04, glibc 2.39, Python 3.13, `gcc` present; datasets mount at `/kaggle/input/datasets/<owner>/<slug>` (run 43115533477). The build container (Ubuntu 22.04, glibc 2.34 symbols) runs there |
| libgomp | `bundle.sh` ships the build host's `libgomp.so.*` flat in the bundle (Kaggle datasets skip directories); `run.py` puts it on `LD_LIBRARY_PATH` |
| Auth | `kaggle` 2.2.5 tries `KAGGLE_API_TOKEN` first, then `KAGGLE_USERNAME` + `KAGGLE_KEY` (`kaggle_api_extended.py` `authenticate`) |

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
