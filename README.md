# ir

Local semantic search engine for markdown knowledge bases. Rust port of [qmd](https://github.com/tobi/qmd) with three key differences:

- **Per-collection SQLite** — each collection is an independent file; no shared global index
- **Persistent daemon** — models stay loaded between queries; first search auto-starts it
- **Dual LLM cache** — expander outputs and reranker scores are persisted; repeated queries are instant

Search quality is benchmarked on 4 BEIR datasets; reranking adds up to +14.5% nDCG@10 over pure vector search.

## Features

- **Hybrid search** — BM25 probe → score fusion (0.80·vec + 0.20·bm25) → LLM reranking
- **Query expansion** — typed sub-queries (lex/vec/hyde) when expander model is present
- **Strong-signal shortcut** — skips expansion when top BM25 score ≥ 0.85 with gap ≥ 0.15
- **Daemon mode** — keeps models warm between queries; auto-starts on first search, eliminates per-call model load overhead
- **Dual LLM cache** — expander outputs cached globally (`~/.config/ir/expander_cache.sqlite`); reranker scores cached per-collection; repeated queries skip all inference
- **Per-collection SQLite** — independent WAL journals, isolated backup, zero cross-collection contention
- **Content-addressed storage** — identical files deduplicated by SHA-256 within a collection
- **FTS5 injection-safe** — all user input escaped before FTS5 query construction
- **Metal GPU** — all layers offloaded to Metal on macOS by default; `IR_GPU_LAYERS=N` to override
- **Auto-download** — models fetched from HuggingFace Hub on first use; `HF_HUB_OFFLINE=1` to disable

## Installation

```bash
cargo install --path .
```

Requires Rust 1.80+. On macOS, links llama.cpp with Metal automatically.

## Models

Models are downloaded automatically from HuggingFace Hub on first use and cached in `~/.cache/huggingface/`. No manual setup required.

| Model | HF Repo | Required for |
|---|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | `ggml-org/embeddinggemma-300M-GGUF` | `ir embed`, vector search, hybrid |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | `unsloth/Qwen3.5-0.8B-GGUF` | unified expand + rerank (optional) |
| [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | `unsloth/Qwen3.5-2B-GGUF` | unified expand + rerank (optional) |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` | reranking only (optional) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | `tobil/qmd-query-expansion-1.7B` | expansion only (optional) |

BM25 search works without any models. When `IR_QWEN_MODEL` is set (or a Qwen3.5 GGUF is found in `~/local-models/`), it is used for both expansion and reranking, replacing the two separate models.

To use local models, point to a directory or individual files:

```bash
export IR_MODEL_DIRS="$HOME/my-models"          # ':'-separated list
export IR_QWEN_MODEL="$HOME/local-models/Qwen3.5-2B-Q4_K_M.gguf"   # unified
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

Local search order: env overrides → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/`. If no local file is found, auto-download kicks in. Set `HF_HUB_OFFLINE=1` to disable network access.

Compatibility aliases: `QMD_EMBEDDING_MODEL`, `QMD_RERANKER_MODEL`, `QMD_EXPANDER_MODEL`, `QMD_MODEL_DIRS`.

### GPU

All model layers are offloaded to Metal by default on macOS. To override:

```bash
IR_GPU_LAYERS=0 ir search "query"   # force CPU
IR_GPU_LAYERS=32 ir search "query"  # partial offload
```

## Usage

### Add a collection

```bash
ir collection add notes ~/notes
ir collection add code  ~/code
```

### Index and embed

```bash
ir update                    # index all collections
ir update notes              # one collection
ir update notes --force      # full re-index from scratch

ir embed                     # embed all unembedded documents
ir embed notes --force       # re-embed everything
```

### Search

```bash
ir search "memory safety in rust"
ir search "sqlite architecture" --mode bm25
ir search "async patterns"     --mode vector
ir search "error handling"     --mode hybrid -c notes --min-score 0.4

# Output formats
ir search "ownership" --json
ir search "ownership" --md
ir search "ownership" --files   # paths only
```

### Daemon

```bash
ir daemon start              # start background daemon (auto-started on first search)
ir daemon stop
ir daemon status
```

The daemon keeps models warm in memory. The first query auto-starts it. Subsequent queries over the Unix socket skip model loading entirely.

### Other

```bash
ir status                    # index health per collection
ir collection ls             # list collections
ir collection rm notes       # remove collection
```

## Search Pipeline

```
Query → BM25 probe → score fusion (0.80·vec + 0.20·bm25) → reranking
```

Strong-signal shortcut (BM25 score ≥ 0.85, gap ≥ 0.10) skips all LLM work.
With expander: expand → lex/vec/hyde sub-queries → RRF → rerank top-20.
All LLM outputs cached in SQLite — repeated queries skip inference entirely.

See [research/pipeline.md](research/pipeline.md) for staged async daemon design.

## Benchmark: BEIR (4 datasets, nDCG@10)

EmbeddingGemma 300M embeddings + qmd-expander-1.7B + Qwen3-Reranker-0.6B.

| Dataset | BM25 | Vector | Hybrid | +Reranker | LLM gain |
|---|---|---|---|---|---|
| NFCorpus (323q) | 0.2046 | 0.3898 | 0.3954 | **0.4001** | +1.2% |
| SciFact (300q) | 0.0500 | 0.7847 | 0.7873 | **0.7797** | −1.0% |
| FiQA (648q) | 0.0298 | 0.4324 | 0.4266 | **0.4567** | +7.1% |
| ArguAna (1406q) | 0.0012 | 0.4264 | 0.4263 | **0.4879** | +14.5% |

BM25 fusion (0.80·vec + 0.20·bm25) provides no statistically significant lift over pure vector on any dataset (paired t-test). Reranker gains are largest on conversational/argument retrieval tasks.

See [research/experiment.md](research/experiment.md) for reproduction steps.

## vs qmd

ir is a Rust port of [qmd](https://github.com/tobi/qmd) with a different storage model and a persistent daemon.

| | qmd | ir |
|---|---|---|
| Storage | Single SQLite for all collections | Per-collection SQLite — `rm name.sqlite` to delete |
| Concurrent writes | Shared WAL journal | Independent WAL per collection |
| sqlite-vec | Dynamically loaded `.so` | Statically compiled in |
| Process model | Spawns per query | Daemon keeps models warm |
| LLM cache | Reranker scores (per-collection) | Reranker scores + expander outputs (global) |
| Quality (NFCorpus nDCG@10) | No published numbers | 0.4001 |

### Performance (macOS M4 Max, same models and query)

| | ir | qmd | Ratio |
|---|---:|---:|---|
| **Cold** (no cache) | 3.0s | 9.5s | **3×** |
| **Warm** (daemon + caches hot) | 30ms | 840ms | **28×** |

Same GGUF models (`qmd-query-expansion-1.7B` + `qwen3-reranker-0.6b`) — LLM inference time is identical. Cold difference: ir caps reranking at 20 candidates vs qmd's 40. Warm difference: qmd pays ~800ms process spawn + JS runtime per invocation; ir's daemon round-trip is 30ms (embed + kNN only).

