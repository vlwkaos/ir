# ir

Local semantic search engine for markdown knowledge bases. Rust port of [qmd](https://github.com/tobi/qmd) with a per-collection SQLite architecture and a hybrid search pipeline benchmarked on BEIR/NFCorpus.

## Features

- **Hybrid search** — BM25 probe → score fusion (0.80·vec + 0.20·bm25) → LLM reranking
- **Query expansion** — typed sub-queries (lex/vec/hyde) when expander model is present
- **Strong-signal shortcut** — skips expansion when top BM25 score ≥ 0.85 with gap ≥ 0.15
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
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` | hybrid reranking (optional) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | `tobil/qmd-query-expansion-1.7B` | hybrid query expansion (optional) |

BM25 search works without any models.

To use local models instead, point to a directory or individual files:

```bash
export IR_MODEL_DIRS="$HOME/my-models"          # ':'-separated list
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

### Other

```bash
ir status                    # index health per collection
ir get path/to/doc.md        # fetch document by path
ir collection ls             # list collections
ir collection rm notes       # remove collection
```

## Search Pipeline

```
Query
  │
  ├─ BM25 probe ──► score ≥ 0.85 AND gap ≥ 0.15? ──► return immediately
  │
  ├─ With expander model:
  │    expand → lex/vec/hyde sub-queries
  │    retrieve BM25 + vector per sub-query
  │    RRF fusion  (lex=1.0, vec=1.5, hyde=1.0)
  │
  ├─ Without expander model:
  │    BM25 list + vector list
  │    score fusion: 0.80·vec + 0.20·bm25  ← tuned on NFCorpus
  │
  └─ Reranker (optional): final = 0.40·fused + 0.60·P(relevant)
```

## Benchmark: BEIR/NFCorpus

3,633 medical documents · 323 test queries · graded relevance

| Mode | nDCG@10 | Recall@10 |
|---|---|---|
| BM25 | 0.2037 | 0.0932 |
| Vector | 0.3866 | 0.1926 |
| **Hybrid (score-fusion α=0.80)** | **0.3924** | **0.1952** |

Hybrid beats vector by +1.5% and BM25 by +92.7% on nDCG@10. The α=0.70–0.95 range forms a flat plateau; 0.80 is the robust midpoint. The old pure-RRF approach scored 0.372.

Reproduce:
```bash
# Download dataset (~100MB)
curl -L https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip -o /tmp/nfcorpus.zip
unzip /tmp/nfcorpus.zip -d test-data/

cargo run --release --bin eval -- --data test-data/nfcorpus --mode all
```

## vs qmd

ir implements the same architecture as [qmd](https://github.com/tobi/qmd) with the following differences:

### Database

| | qmd | ir |
|---|---|---|
| Storage | Single `~/.cache/qmd/index.sqlite` for all collections | Per-collection `~/.local/share/ir/collections/<name>.sqlite` |
| Collection isolation | `collection` column + `UNIQUE(collection, path)` | Filesystem-level — the DB file *is* the collection |
| Concurrent writes | All collections share one WAL journal | Independent WAL per collection, no contention |
| Delete a collection | `DELETE WHERE collection = ?` (file stays) | `rm name.sqlite` — complete, portable |
| FTS5 scope | Index spans all collections, filtered per query | Index scoped to one collection — smaller, faster |
| sqlite-vec | Dynamically loaded `.so` | Statically compiled in |
| Deduplication | Content-addressed across all collections | Content-addressed within each collection |

### Search

| | qmd | ir |
|---|---|---|
| Strong-signal threshold | ≥ 0.85, gap ≥ 0.15 | same |
| Fusion (no expander) | RRF (bm25=1.0, vec=1.2) | Score fusion 0.80·vec + 0.20·bm25 |
| Fusion (with expander) | RRF (bm25=1.0, vec=1.2, reranker=2.5) | RRF (lex=1.0, vec=1.5, hyde=1.0) |
| Reranker blend | Position-based (75/25 top, 60/40 mid, 40/60 low) | Fixed: 0.40·fusion + 0.60·reranker |
| BM25 normalization | Sigmoid `1/(1+exp(-(abs-5)/3))` | `(-raw)/(1+(-raw))` |
| Benchmarked | No published numbers | NFCorpus nDCG@10 = 0.393 |

### Performance (macOS M4 Max)

| Mode | ir | qmd (Bun) | Ratio |
|---|---|---|---|
| BM25 (no model) | 341 ms | 381 ms | 1.1× |
| Vector search | **570 ms** | 3,901 ms | **6.8×** |

BM25 is close because Bun is fast and SQLite does the work. Vector search is 6.8× faster because ir links llama.cpp natively at compile time — no FFI startup overhead per invocation. For repeated searches in a session (10 vector queries): ir ~5.7s vs qmd ~39s.

## Development

```bash
cargo build                  # debug build
cargo build --release        # release build
cargo test                   # unit tests (55 passing, no models required)
cargo test -- --ignored      # model-dependent tests (requires models)
cargo run --bin eval -- --data test-data/nfcorpus --mode all
```

## Schema

Each collection database has 8 objects:

```
content          — hash → full text (content-addressed)
documents        — path, title, hash, active flag
documents_fts    — FTS5 virtual table (porter tokenizer)
vectors_vec      — sqlite-vec kNN (768d cosine, EmbeddingGemma format)
content_vectors  — chunk metadata (hash, seq, pos, model)
llm_cache        — reranker score cache (query+doc hash → score)
meta             — collection metadata (name, schema version)
```

Triggers keep `documents_fts` in sync with `documents` on insert/update/delete.
