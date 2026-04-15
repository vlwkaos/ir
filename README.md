# ir

[ENG](README.md) | [한국어](README.ko.md)

Local semantic search engine for markdown knowledge bases. Rust port of [qmd](https://github.com/tobi/qmd) with three key differences:

- **Per-collection SQLite** — each collection is an independent file; no shared global index
- **Persistent daemon** — models stay loaded between queries; first search auto-starts it
- **Dual LLM cache** — expander outputs and reranker scores are persisted; repeated queries are instant

Search quality benchmarked on 4 BEIR datasets; reranking adds up to +14.5% nDCG@10 over pure vector.

<details>
<summary><strong>Features</strong></summary>

- **Hybrid search** — BM25 probe → score fusion (0.80·vec + 0.20·bm25) → LLM reranking
- **Query expansion** — typed sub-queries (lex/vec/hyde) when expander model is present
- **Strong-signal shortcut** — skips expansion when top BM25 score ≥ 0.75 with gap ≥ 0.10
- **Daemon mode** — keeps models warm between queries; auto-starts on first search
- **Dual LLM cache** — expander outputs cached globally; reranker scores cached per-collection
- **Per-collection SQLite** — independent WAL journals, isolated backup, zero cross-collection contention
- **Content-addressed storage** — identical files deduplicated by SHA-256 within a collection
- **FTS5 injection-safe** — all user input escaped before FTS5 query construction
- **Metal GPU** — all layers offloaded to Metal on macOS by default; `IR_GPU_LAYERS=N` to override
- **Auto-download** — models fetched from HuggingFace Hub on first use; `HF_HUB_OFFLINE=1` to disable

</details>

## Installation

**Homebrew (macOS):**

```bash
brew install vlwkaos/tap/ir
```

**From source:**

```bash
cargo install --path .
```

Requires Rust 1.80+. On macOS, links llama.cpp with Metal automatically.

## Quick start

```bash
ir collection add notes ~/notes   # register a collection
ir update notes                   # scan files → extract text → populate FTS5 index (BM25)
ir embed notes                    # chunk text → run embedding model → store vectors (enables vector + hybrid search)
ir search "memory safety in rust" # search (daemon auto-starts)
```

`ir update` is fast (no models, pure text processing). `ir embed` is slow on first run (model inference per chunk) but only re-embeds changed content on subsequent runs. BM25 search works after `update` alone; vector and hybrid search require `embed`.

<details>
<summary><strong>Models</strong></summary>

Models are downloaded automatically from HuggingFace Hub on first use and cached in `~/.cache/huggingface/`. No manual setup required.

| Model | HF Repo | Required for |
|---|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | `ggml-org/embeddinggemma-300M-GGUF` | `ir embed`, vector search, hybrid |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | `unsloth/Qwen3.5-0.8B-GGUF` | unified expand + rerank (optional) |
| [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | `unsloth/Qwen3.5-2B-GGUF` | unified expand + rerank (optional) |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` | reranking only (optional) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | `tobil/qmd-query-expansion-1.7B` | expansion only (optional) |
| [BGE-M3 568M](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) | `ggml-org/bge-m3-Q8_0-GGUF` | Korean embedding alternative (optional) |

BM25 search works without any models. When `IR_QWEN_MODEL` is set (or a Qwen3.5 GGUF is found in `~/local-models/`), it replaces both the expander and reranker.

**Local models:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_QWEN_MODEL="$HOME/local-models/Qwen3.5-2B-Q4_K_M.gguf"   # unified
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

Search order: env → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/` → HF Hub auto-download.

`IR_*_MODEL` env vars accept a path to a `.gguf` file, a directory containing a known model file, or a HuggingFace repo ID (`owner/name`). Unrecognized values error immediately instead of silently loading the default.

Known HF repo IDs: `ggml-org/embeddinggemma-300M-GGUF`, `ggml-org/bge-m3-Q8_0-GGUF`, `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`, `tobil/qmd-query-expansion-1.7B`, `unsloth/Qwen3.5-0.8B-GGUF`, `unsloth/Qwen3.5-2B-GGUF`.

Compatibility aliases: `QMD_EMBEDDING_MODEL`, `QMD_RERANKER_MODEL`, `QMD_EXPANDER_MODEL`, `QMD_MODEL_DIRS`.

**GPU:**

```bash
IR_GPU_LAYERS=0 ir search "query"   # force CPU
IR_GPU_LAYERS=32 ir search "query"  # partial offload
```

</details>

<details>
<summary><strong>Usage</strong></summary>

**Collections:**

```bash
ir collection add notes ~/notes
ir collection add code  ~/code
ir collection ls
ir collection rm notes
ir status                    # index health per collection
```

**Index and embed:**

```bash
ir update                    # index all collections
ir update notes              # one collection
ir update notes --force      # full re-index from scratch

ir embed                     # embed all unembedded documents
ir embed notes --force       # re-embed everything
```

**Search:**

```bash
ir search "memory safety in rust"
ir search "sqlite architecture" --mode bm25
ir search "async patterns"     --mode vector
ir search "error handling"     --mode hybrid -c notes --min-score 0.4

# Output formats
ir search "ownership" --json
ir search "ownership" --md
ir search "ownership" --files       # paths only
ir search "ownership" --full        # include full document content in results
ir search "ownership" --chunk       # include best-matching chunk text (vector results)
```

**Retrieve documents:**

```bash
ir get "2026/Daily/04/2026-04-07.md"           # collection-relative path
ir get "Notes/2026/Daily/04/2026-04-07.md"     # vault-root path (strips collection dir prefix)
ir get "2026-04-07" -c periodic                # substring match, scoped to collection
ir get "some/path.md" --json                   # full metadata as JSON
ir get "some/path.md" --section "Installation" # extract named heading section only
ir get "some/path.md" --max-chars 3000         # first 3000 chars
ir get "some/path.md" --offset 1000 --max-chars 2000  # chars 1000–3000

ir multi-get "file1.md" "file2.md" "file3.md"  # batch fetch
ir multi-get "file1.md" "file2.md" --json       # {found: [...], not_found: [...]}
ir multi-get "file1.md" "file2.md" --files      # paths only (found ones)
ir multi-get "file1.md" "file2.md" --max-chars 2000  # truncate each doc
```

Path matching order: exact → suffix (`%/path`) → substring. Vault-root paths (where the first component matches the collection's directory name) are resolved before the normal match.

**Daemon:**

```bash
ir daemon start              # start (auto-started on first search)
ir daemon stop
ir daemon status
```

The daemon keeps models warm in memory. Subsequent queries over the Unix socket skip model loading entirely (~30ms round-trip vs 3s cold).

</details>

<details>
<summary><strong>MCP server — Claude Desktop / Claude Code</strong></summary>

`ir mcp` runs a Model Context Protocol server so Claude can search your indexed documents directly.

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ir": {
      "command": "ir",
      "args": ["mcp"]
    }
  }
}
```

**Claude Code** (`.mcp.json` in project root or `~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "ir": {
      "command": "ir",
      "args": ["mcp"]
    }
  }
}
```

Five tools are exposed:

| Tool | Description |
|------|-------------|
| `search` | Hybrid BM25+vector search. Returns path, title, score, snippet. Params: `mode`, `limit`, `min_score`, `collections`, `full` (include full doc text), `include_chunk` (include best-matching chunk text). |
| `get` | Retrieve document text by path (exact → suffix → substring match). Params: `collections`, `section` (heading text, case-insensitive), `offset` (char offset), `max_chars` (truncate). |
| `multi_get` | Batch document retrieval. Params: `paths[]`, `collections`, `max_chars` (truncate each doc). Returns `found` and `not_found`. |
| `status` | Index health — collection names, doc counts, DB sizes, daemon status. |
| `update` | Re-index collections after file changes. Accepts `collection` and `force` params. |

**HTTP mode** (for remote access or multi-client setups):

```bash
ir mcp --http 3620    # serve on all interfaces, port 3620
```

Configure clients to point at `http://<host>:3620/mcp`. The daemon starts automatically on first search tool call.

> **Security note:** HTTP mode is unauthenticated and binds to all interfaces. Only expose it on trusted networks. The `update` tool can trigger re-indexing, so treat it like any other local write-access service.

</details>

<details>
<summary><strong>Preprocessors — Korean / Japanese / Chinese</strong></summary>

Preprocessors tokenize text before BM25 indexing. Without one, agglutinated words ("이스탄불의", "東京都") are treated as single FTS tokens and never match morpheme-level queries. The same preprocessor runs at index time and query time.

**Korean (lindera, Mode::Decompose):**

```bash
ir preprocessor install ko          # downloads lindera-tokenize, registers as "ko"
                                    # shows collection picker to bind immediately
ir collection add wiki ~/wiki       # add collection (if not yet added)
ir preprocessor bind ko wiki        # wire "ko" to collection and re-index
ir search "서울 지하철" -c wiki
```

`ir preprocessor install ko` downloads a pre-built binary from the GitHub release — embedded ko-dic dictionary, no system deps, no Rust toolchain required. Supported platforms: **macOS** (universal binary, Intel + Apple Silicon) and **Linux** (x86\_64, aarch64). The install step shows an interactive picker so you can bind to collections right away.

**Other languages:**

```bash
ir preprocessor install ja    # Japanese (Lindera + ipadic)
ir preprocessor install zh    # Chinese (bigram tokenizer)
```

**Manage:**

```bash
ir preprocessor list          # shows registered + available bundled preprocessors
ir preprocessor remove ko     # unregister (keeps binary)
ir preprocessor remove ko -d  # unregister and delete binary
```

The protocol is stdin/stdout line-by-line: one UTF-8 line in, one tokenized line out, process stays alive between lines. Any executable following this protocol can be registered.

Lindera throughput: ~5,600 Korean docs/s · 1.8 MB/s on M-series Mac. Near-zero cold start (Rust binary, embedded dictionary).

**Korean BM25 benchmark** (MIRACL-Korean, 213 queries):

| preprocessor | nDCG@10 | note |
|---|---|---|
| none | 0.0009 | agglutinated tokens never match |
| lindera | 0.0460 | 50× gain from morphological tokenization |
| lindera hybrid+rerank | **0.8411** | near-ceiling on 2,835 passages |

Compound decompounding benchmark (50 queries targeting compound sub-components):

| preprocessor | nDCG@10 | note |
|---|---|---|
| none | 0.0000 | sub-parts absent from FTS index |
| lindera | **0.6326** | Mode::Decompose splits compounds |

See [research/experiment.md](research/experiment.md) for full results and rationale.

**Korean embedding models**: For Korean-optimized dense retrieval, [BGE-M3](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) can replace the default embedding model via `IR_EMBEDDING_MODEL`. Filename auto-detection handles pooling and formatting. See [README.ko.md](README.ko.md) for setup. Switching models requires `ir embed --force` (vector dimensions auto-adapt).

</details>

<details>
<summary><strong>Search Pipeline</strong></summary>

```
Query → BM25 probe → score fusion (0.80·vec + 0.20·bm25) → reranking
```

Strong-signal shortcut (BM25 score ≥ 0.75, gap ≥ 0.10) skips all LLM work.
With expander: expand → lex/vec/hyde sub-queries → RRF → rerank top-20.
All LLM outputs cached in SQLite — repeated queries skip inference entirely.

See [research/pipeline.md](research/pipeline.md) for staged async daemon design.

</details>

<details>
<summary><strong>Benchmark — BEIR (4 datasets, nDCG@10)</strong></summary>

EmbeddingGemma 300M embeddings + qmd-expander-1.7B + Qwen3-Reranker-0.6B.

| Dataset | BM25 | Vector | Hybrid | +Reranker | LLM gain |
|---|---|---|---|---|---|
| NFCorpus (323q) | 0.2046 | 0.3898 | 0.3954 | **0.4001** | +1.2% |
| SciFact (300q) | 0.0500 | 0.7847 | 0.7873 | **0.7797** | −1.0% |
| FiQA (648q) | 0.0298 | 0.4324 | 0.4266 | **0.4567** | +7.1% |
| ArguAna (1406q) | 0.0012 | 0.4264 | 0.4263 | **0.4879** | +14.5% |

BM25 fusion provides no statistically significant lift over pure vector (paired t-test). Reranker gains are largest on conversational/argument retrieval.

See [research/experiment.md](research/experiment.md) for reproduction steps.

</details>

<details>
<summary><strong>vs qmd</strong></summary>

ir is a Rust port of [qmd](https://github.com/tobi/qmd) with a different storage model and a persistent daemon.

| | qmd | ir |
|---|---|---|
| Storage | Single SQLite for all collections | Per-collection SQLite — `rm name.sqlite` to delete |
| Concurrent writes | Shared WAL journal | Independent WAL per collection |
| sqlite-vec | Dynamically loaded `.so` | Statically compiled in |
| Process model | Spawns per query | Daemon keeps models warm |
| LLM cache | Reranker scores (per-collection) | Reranker scores + expander outputs (global) |
| Quality (NFCorpus nDCG@10) | No published numbers | 0.4001 |

**Performance** (macOS M4 Max, same models and query):

| | ir | qmd | Ratio |
|---|---:|---:|---|
| **Cold** (no cache) | 3.0s | 9.5s | **3×** |
| **Warm** (daemon + caches hot) | 30ms | 840ms | **28×** |

Cold difference: ir caps reranking at 20 candidates vs qmd's 40. Warm difference: qmd pays ~800ms process spawn + JS runtime per invocation; ir's daemon round-trip is 30ms (embed + kNN only).

</details>

<details>
<summary><strong>Development</strong></summary>

```bash
cargo build                  # debug build
cargo build --release        # release build
cargo test                   # unit tests (no models required)
cargo test -- --ignored      # model-dependent tests (requires models)
cargo run --bin eval -- --data test-data/nfcorpus --mode all
```

</details>

<details>
<summary><strong>Schema</strong></summary>

Each collection database (`~/.config/ir/collections/<name>.sqlite`):

```
content          — hash → full text (content-addressed)
documents        — path, title, hash, active flag
documents_fts    — FTS5 virtual table (porter tokenizer)
vectors_vec      — sqlite-vec kNN (768d cosine, EmbeddingGemma format)
content_vectors  — chunk metadata (hash, seq, pos, model)
llm_cache        — reranker score cache (sha256(model+query+doc) → score)
meta             — collection metadata (name, schema version)
```

Global cache (`~/.config/ir/expander_cache.sqlite`):

```
expander_cache   — sha256(model+query) → JSON Vec<SubQuery>
```

Triggers keep `documents_fts` in sync with `documents` on insert/update/delete.

</details>
