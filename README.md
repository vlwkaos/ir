# ir

[ENG](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

Local semantic search engine for Markdown knowledge bases and mixed code repositories. Rust port of [qmd](https://github.com/tobi/qmd) with three key differences:

- **Per-collection SQLite** — each collection is an independent file; no shared global index
- **Persistent daemon** — models stay loaded between queries; first search auto-starts it
  Cold first queries can still return BM25 immediately while the daemon warms in the background.
- **Dual LLM cache** — expander outputs and reranker scores are persisted; repeated queries are instant
- **Linked retrieval** — optional code-aware units plus explicit Markdown/comment links for agent context

Search quality benchmarked on 4 BEIR datasets; reranking adds up to +14.5% nDCG@10 over pure vector.

<details>
<summary><strong>Features</strong></summary>

- **Hybrid search** — BM25 probe → score fusion (0.80·vec + 0.20·bm25) → LLM reranking
- **Query expansion** — typed sub-queries (lex/vec/hyde) when expander model is present
- **Strong-signal shortcut** — skips expansion when top BM25 score ≥ 0.75 with gap ≥ 0.10
- **Daemon mode** — keeps models warm between queries; auto-starts on first search
  Cold start does not have to block the first useful BM25 result.
- **Dual LLM cache** — expander outputs cached globally; reranker scores cached per-collection
- **Per-collection SQLite** — independent WAL journals, isolated backup, zero cross-collection contention
- **Content-addressed storage** — identical files deduplicated by SHA-256 within a collection
- **FTS5 injection-safe** — all user input escaped before FTS5 query construction
- **GPU acceleration** — Metal on macOS (default), CUDA/ROCm/Vulkan on Linux (opt-in via feature flags); `IR_GPU_LAYERS=N` to override
- **Auto-download** — models fetched from HuggingFace Hub on first use; `HF_HUB_OFFLINE=1` to disable
- **Linked retrieval** — mixed Markdown/code collections, code-unit snippets, and explicit `--related` context

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

Requires Rust 1.80+. On macOS, links llama.cpp with Metal automatically. On Linux, pass `--features llama-cuda`, `llama-rocm`, or `llama-vulkan` to enable GPU acceleration.

## Quick start

```bash
ir collection add notes ~/notes   # register a collection
ir update notes                   # scan files → extract text → populate FTS5 index (BM25)
ir embed notes                    # chunk text → run embedding model → store vectors (enables vector + hybrid search)
ir search "memory safety in rust" # search (daemon auto-starts)
```

`ir update` is fast (no models, pure text processing). `ir embed` is slow on first run (model inference per chunk) but only re-embeds changed content on subsequent runs. BM25 search works after `update` alone; vector and hybrid search require `embed`.

## Linked retrieval for code + knowledge

Markdown-only behavior is unchanged. To opt into mixed code/knowledge retrieval, register a collection with the built-in preset:

```bash
ir collection add project . --preset mixed
ir update project
ir embed project

ir search "why does hybrid search skip reranking" \
  -c project \
  --chunk \
  --related 3 \
  --json
```

`--preset mixed` includes Markdown plus mainstream code extensions (Rust, Python, JS/TS, Go, Java, C/C++, C#, Ruby, PHP, Swift, Kotlin, Scala, shells, Lua, Dart, Elixir, Erlang, F#, Clojure) and excludes common build/vendor directories. Code extraction is best-effort: common symbols become indexed units; unsupported code shapes still index as file-level units.

Related links are deterministic, not inferred. `ir update` extracts:

```md
---
related:
  - src/search/hybrid.rs#is_bm25_strong_signal
---

See [[Search pipeline#Strong signal shortcut]] and [strong-signal-shortcut].
```

```rust
// [strong-signal-shortcut]
// See [[Search pipeline#Strong signal shortcut]].
fn is_bm25_strong_signal(...) -> bool { ... }
```

Line ranges are display hints. Search results include indexed unit text, `indexed_hash` (the indexed unit text hash), symbol/range metadata, markers, and one-hop `related` items; live file navigation should treat a changed file as stale until `ir update` runs again. See [docs/linked-retrieval-agent.md](docs/linked-retrieval-agent.md) for the minimal agent writing/search contract.

Linked retrieval has a model-free synthesis evidence fixture:

```bash
python3 scripts/linked-synthesis-eval.py --ir-bin target/debug/ir
```

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

BM25 search works without any models. The default tier-2 path is the dedicated expander + reranker. `IR_COMBINED_MODEL` is opt-in for explicit combined-model experiments or testing.

**Local models:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

Combined mode is explicit-only:

```bash
export IR_COMBINED_MODEL="$HOME/local-models/Qwen3.5-2B-Q4_K_M.gguf"   # testing / experiments only
```

Search order: env → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/` → HF Hub auto-download.

`IR_*_MODEL` env vars accept a path to a `.gguf` file, a directory containing a known model file, or a HuggingFace repo ID (`owner/name`). Unrecognized values error immediately instead of silently loading the default.

Known HF repo IDs: `ggml-org/embeddinggemma-300M-GGUF`, `ggml-org/bge-m3-Q8_0-GGUF`, `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`, `tobil/qmd-query-expansion-1.7B`, `unsloth/Qwen3.5-0.8B-GGUF`, `unsloth/Qwen3.5-2B-GGUF`.

Compatibility aliases: `QMD_EMBEDDING_MODEL`, `QMD_RERANKER_MODEL`, `QMD_EXPANDER_MODEL`, `QMD_MODEL_DIRS`.

**Config directory:**

```bash
export IR_CONFIG_DIR="~/vault/.config/ir"   # portable across machines
```

`IR_CONFIG_DIR` sets the directory for config, collection DBs, and daemon files. Supports `~` and `$VAR` expansion, so the value is safe to use in MCP configs synced across machines. Precedence: `IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir` (deprecated) → `~/.config/ir`.

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
ir search "ownership" --chunk       # include best-matching chunk/unit text
ir search "ownership" --related 3   # include explicit one-hop linked context (max 20)
ir search "ownership" --quiet       # suppress stderr (progress, logs) — for scripting

# Filter by field (-f/--filter, repeatable; all clauses ANDed)
ir search "design" -f "modified_at>=2026-01-01"
ir search "design" -f "meta.tags=rust"
ir search "design" -f "path~notes/"
ir search "design" -f "modified_at>=2025-01-01" -f "meta.author=vlwkaos"
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

**Filter syntax (`-f/--filter`):**

Each clause is a string `FIELD OP VALUE`. Multiple `-f` flags are ANDed together.

| Field | Description |
|-------|-------------|
| `path` | Document path (relative to collection root) |
| `modified_at` | File modification time (UTC RFC3339) |
| `created_at` | File creation time (UTC RFC3339) |
| `meta.<name>` | Frontmatter field (e.g. `meta.tags`, `meta.author`) |

| Op | Meaning |
|----|---------|
| `=` / `!=` | Equal / not equal (case-sensitive) |
| `>` / `>=` / `<` / `<=` | Lexicographic compare (dates normalize to UTC RFC3339) |
| `~` / `!~` | Contains / not-contains (case-insensitive) |

Date values for `modified_at`, `created_at`, and `meta.date` are normalized to UTC RFC3339 (`YYYY-MM-DD` becomes `YYYY-MM-DDT00:00:00Z`). Multi-valued frontmatter fields (e.g. tag arrays) match if **any** element satisfies the clause — including `!=`. A doc tagged `["rust", "go"]` passes `meta.tags!=rust` because `"go"` satisfies the condition. Documents with no metadata rows always fail `meta.*` clauses.

> **Note:** Collection DBs are upgraded automatically on first use. Schema v2 backfills frontmatter metadata; schema v3 backfills indexed units and explicit links. Both migrations are local and preserve existing Markdown search behavior.

**Daemon:**

```bash
ir daemon start              # start (auto-started on first search)
ir daemon stop
ir daemon status
```

The daemon keeps models warm in memory. Subsequent queries over the Unix socket skip model loading entirely (~30ms round-trip vs 3s cold). On a cold start, `ir search` kicks off daemon startup immediately; if BM25 already found a usable answer, that first query can return the BM25 result while the daemon continues warming in the background.

</details>

<details>
<summary><strong>Incremental Indexing</strong></summary>

IR efficiently handles updates by only processing changed files through content-addressed storage with SHA-256 hashing.

**How it works:**

- **Change detection**: Files are hashed (SHA-256) and compared against stored hashes
- **Smart updates**: Only modified or new files are re-processed
- **Deletion handling**: Removed files are marked as inactive (soft delete)
- **Deduplication**: Identical content within a collection shares storage

**Index operations:**

```bash
# Regular incremental update (default)
ir update                    # all collections
ir update notes              # specific collection

# Force full re-index from scratch
ir update notes --force      # rebuilds entire index

# Check what changed (see the summary)
ir update notes
# Output: "2 added, 1 updated, 0 deactivated"
```

**Embedding operations:**

```bash
# Incremental embedding (only new/changed documents)
ir embed                     # embeds unembedded content
ir embed notes               # specific collection

# Force re-embedding everything
ir embed notes --force       # re-computes all vectors
```

**Performance characteristics:**

- Initial indexing: fast (no models, pure text extraction)
- Incremental updates: only processes changed files
- Hash comparison: instant even for thousands of files
- Embedding: slow first time, fast incremental updates

**Example workflow:**

```bash
# Monday: initial setup
ir collection add notes ~/notes
ir update notes              # indexes 500 files
ir embed notes               # computes 500 embeddings (slow)

# Tuesday: added 3 files, modified 2
ir update notes              # Output: "3 added, 2 updated, 0 deactivated"
ir embed notes               # only embeds 5 documents (fast)

# Wednesday: deleted 1 file
ir update notes              # Output: "0 added, 0 updated, 1 deactivated"
# No embedding needed for deletions
```

The incremental approach means you can run `ir update` frequently without performance penalty — only changed content is processed.

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
| `search` | Hybrid BM25+vector search. Returns path, title, score, snippet. Params: `mode`, `limit`, `min_score`, `collections`, `full` (include full doc text), `include_chunk` (include best-matching chunk/unit text), `include_related`, `related_limit` (max 20), `filter` (array of `{field, op, value}` objects, ANDed). |
| `get` | Retrieve document text by path (exact → suffix → substring match). Params: `collections`, `section` (heading text, case-insensitive), `offset` (char offset), `max_chars` (truncate). |
| `multi_get` | Batch document retrieval. Params: `paths[]`, `collections`, `max_chars` (truncate each doc). Returns `found` and `not_found`. |
| `status` | Index health — collection names, doc counts, DB sizes, daemon status. |
| `update` | Re-index collections after file changes. Accepts `collection` and `force` params. |

The `filter` array accepts structured clauses: `{"field": "modified_at", "op": ">=", "value": "2024-01-01"}`. Fields: `path`, `modified_at`, `created_at`, `meta.<name>`. Ops: `=`, `!=`, `>`, `>=`, `<`, `<=`, `~` (contains), `!~` (not-contains).

**HTTP mode** (for remote access or multi-client setups):

```bash
ir mcp --http 3620                              # serve on all interfaces, port 3620
ir mcp --http 3620 --cors '*'                   # allow any browser origin (wildcard)
ir mcp --http 3620 --cors 'https://app.example.com'  # allow specific origin only
```

Configure clients to point at `http://<host>:3620/mcp`. The daemon starts automatically on first search tool call.

`--cors` sets `Access-Control-Allow-Origin` so browser-hosted clients (web apps, Claude.ai web) can connect. `--cors '*'` also disables rmcp's DNS-rebinding host check; use only on trusted networks. Omitting `--cors` emits no CORS headers (curl/CLI clients unaffected).

> **Security note:** HTTP mode is unauthenticated and binds to all interfaces. Only expose it on trusted networks. The `update` tool can trigger re-indexing, so treat it like any other local write-access service.

</details>

<details>
<summary><strong>Preprocessors — Korean / Japanese / Chinese</strong></summary>

Preprocessors tokenize text before BM25 indexing. Without one, agglutinated words ("이스탄불의", "東京都") are treated as single FTS tokens and never match morpheme-level queries. The same preprocessor runs at index time and query time.

**Korean (lindera, Mode::Decompose):**

```bash
ir preprocessor install ko          # downloads official lindera CLI + ko-dic, registers as "ko"
                                    # shows collection picker to bind immediately
ir collection add wiki ~/wiki       # add collection (if not yet added)
ir preprocessor bind ko wiki        # wire "ko" to collection and re-index
ir search "서울 지하철" -c wiki
```

`ir preprocessor install ko` downloads the official lindera CLI binary and ko-dic dictionary from lindera's GitHub releases. Supported platforms: **macOS** (arm64, x86\_64) and **Linux** (x86\_64, aarch64). No system deps, no Rust toolchain required. The install step shows an interactive picker so you can bind to collections right away.
Binding the built-in `ko` alias also writes the current Korean routing default for that collection:

```yaml
routing:
  fused_strong_product: 0.05
```

This is a bind-time default, not a hidden runtime special case. If you already set a `routing:` block yourself, that explicit config wins.

**Per-collection routing overrides** (`config.yml`, optional):

```yaml
collections:
  - name: wiki-ko
    path: ~/wiki
    preprocessor: [ko]
    routing:
      fused_strong_product: 0.05
```

Use this to override BM25/fused strong-signal thresholds for a specific collection. The fields are:

- `fused_strong_floor`
- `fused_strong_product`
- `bm25_strong_floor`
- `bm25_strong_gap`

Overrides apply only when all searched collections agree on the same value. Mixed searches with conflicting overrides fall back to the global default thresholds.

**Japanese:**

```bash
ir preprocessor install ja    # Japanese (Lindera + ipadic)
```

**Chinese (lindera + jieba, Mode::Decompose):**

```bash
ir preprocessor install zh          # downloads lindera CLI + jieba dict, registers as "zh"
ir collection add notes ~/notes     # add collection
ir preprocessor bind zh notes       # wire "zh" to collection and re-index
ir search "机器学习" -c notes
```

`ir preprocessor install zh` downloads the official lindera CLI binary and jieba segmentation dictionary from lindera's GitHub releases (same binary as `ko`/`ja`, different dict). Supported platforms: **macOS** (arm64, x86\_64) and **Linux** (x86\_64, aarch64).

Unlike `ko`, binding `zh` does **not** write a routing override — the global strong-signal thresholds apply. Add a collection-level `routing:` block if you observe too many or too few tier-1 escalations on your corpus.

**Limitation:** No stopword filtering. Function words (的、了、在、是、…) are indexed as terms. BM25 precision is lower on function-word-heavy queries; hybrid+rerank compensates.

**Manage:**

```bash
ir preprocessor list          # shows registered + available bundled preprocessors
ir preprocessor remove ko     # unregister (keeps binary)
ir preprocessor remove ko -d  # unregister and delete binary
```

The protocol is stdin/stdout line-by-line: one UTF-8 line in, zero or one tokenized line out (zero if all tokens are filtered), process stays alive between lines. The subprocess must pass ASCII-only single-word lines through unchanged — `ir` uses an internal sentinel token to detect when a line produces no output. Any executable following this protocol can be registered.

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
`scripts/bench.sh <dataset>` prints a per-mode table (`bm25`, `vector`, `hybrid`) and caches the full JSON under `logs/results/<dataset>/`.
For non-BM25 runs, the benchmark wrapper pins tier-2 to the dedicated expander + reranker path and restarts the benchmark daemon before scoring, so a local Qwen combined GGUF or stale daemon state does not silently change the benchmark pipeline.
If a machine crash happens after prepare/index/embed but before scoring finishes, rerunning the same `scripts/bench.sh <dataset>` command resumes from the prepared collection instead of starting from zero. Query scoring also resumes automatically: `scripts/beir-eval.py run --output ...` writes per-query sidecar progress to `<output>.partial/` and continues from that on rerun.
Large corpora can be benchmarked on a sampled pool with `--size N --seed N`. For MIRACL-Ko, `scripts/bench.sh miracl-ko --size 50000` is the current research default because `10000` docs proved stable but too saturated for meaningful ranking comparisons. The pool-size study still establishes `10000` as the minimum stable sampled benchmark.
Maintainer shortcut: `scripts/research-harness.sh` wraps the supported research flows for baseline locking, signal collection, threshold sweeps, and automated holdout validation of shortlisted thresholds. For fine sweeps near a threshold cliff, `validate-thresholds` also accepts explicit fused values via `--products ...`. See [research/experiment.md](research/experiment.md).
Current research direction:

- keep `fiqa` on the current fused threshold
- use `miracl-ko --size 50000` for Korean threshold research
- prefer stricter fused gating before trying a learned router
- treat router work as offline-only until it clearly beats simple gating on holdout

Tier-2 router research is intentionally separate from that harness. First collect router-grade signals with `bash scripts/signal-sweep.sh --dataset miracl-ko --size 50000 --pools 3 --tier1`, then use `bash scripts/router-data.sh ko` to prepare a Korean-only `smoltrain` bundle. The router benchmark itself also stays offline by default: collect holdout signals for `miracl-ko-s50000-p42`, then score the checkpoint with `scripts/router-bench.py` instead of changing the shipped runtime path.
On macOS, `scripts/bench.sh` now runs long benchmark phases under a safety watchdog by default. Metal stays enabled for speed, but the wrapper aborts the run if free memory drops too low, swapouts begin, or `ir` sustains CPU-fallback-like usage. Tune with `IR_BENCH_MIN_FREE_PCT`, `IR_BENCH_MAX_IR_CPU_PCT`, `IR_BENCH_CPU_STRIKES`, or disable with `IR_BENCH_GUARD=0`.

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
