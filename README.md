# ir

[![crates.io](https://img.shields.io/crates/v/ir-search.svg)](https://crates.io/crates/ir-search)
[![CI - check](https://github.com/vlwkaos/ir/actions/workflows/ci.yml/badge.svg)](https://github.com/vlwkaos/ir/actions/workflows/ci.yml)
[![CI - evaluation](https://github.com/vlwkaos/ir/actions/workflows/ci-evaluation.yml/badge.svg)](https://github.com/vlwkaos/ir/actions/workflows/ci-evaluation.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ENG](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

Local search for markdown knowledge bases, in any mix of languages. BM25 + vector + LLM reranking, entirely on your machine — one SQLite file per collection, models kept warm by a persistent daemon, all LLM outputs cached. Optionally indexes source code beside the notes as supporting evidence.

```bash
brew install vlwkaos/tap/ir          # macOS
cargo install ir-search              # any platform (binary name: ir)
```

```bash
ir collection add notes ~/notes      # register a collection
ir sync notes                        # index text + embed vectors
ir search "memory safety in rust"    # search (daemon auto-starts)
```

BM25 search works with no models at all. Vector/hybrid search downloads models automatically from HuggingFace on first use. Requires Rust 1.80+ if building from source; Metal is linked automatically on macOS, and Linux GPU backends are opt-in (`--features llama-cuda|llama-rocm|llama-vulkan`).

## How it searches — you only pay for what's hard

Three tiers. Each runs **only if the previous one wasn't confident**, so most queries
return from tier 0 or 1 and never touch an LLM. One warm daemon holds every model in
memory; every LLM output is cached in SQLite.

```
query
  │
  ▼  TIER 0 · no model · instant
  BM25 (+ doc-graph neighbours, + documents that matching code links to)
  clear keyword winner? ─── yes ──► results
  │
  ▼  TIER 1 · embedding model · ~50 ms
  BM25  ∥  embed query ─► vector kNN (HNSW)          run in parallel
  fuse, choosing per query:
    a top BM25 passage holds every query word ─► vector + BM25 boost for exact matches
    answers are in another script than the query ─► vector only (cross-language)
    otherwise ─► 0.8 · vector + 0.2 · BM25
  confident? ─── yes ──► results
  │
  ▼  TIER 2 · reranker 0.6B · ~2–5 s
  rerank the top 100 ─► results
```

The per-query choice keeps keyword lookups exact while a question finds its answer
in another language: an English question matches a Korean note, and the other way round.

The preprocessor is the make-or-break stage for Korean/Japanese/Chinese BM25: it
tokenizes both the indexed text and the query, and without it CJK BM25 scores near zero.

**Cold vs warm** (M4 Max): first query ~3.0 s while the daemon loads; every query after,
~30 ms round-trip, and BM25 stays instant even during a cold start. Pipeline knobs are
tunable per collection under [Advanced Configuration](#advanced-configuration).

## Measured quality (nDCG@10)

Each column is the score **if the query stops at that stage**. `BM25` and `Vector` are the single-signal baselines; `Hybrid` is tier 1; `+ Rerank` adds the 0.6B reranker over the top 100 (tier 2). The LLM expander is **off by default**.

| Corpus | BM25 | Vector | Hybrid | + Rerank |
|---|---|---|---|---|
| NFCorpus (en, 3.6k docs, 323 q) | 0.31 | 0.39 | 0.39 | 0.40 |
| FiQA (en, 57.6k docs, 648 q) | 0.24 | 0.40 | 0.40 | **0.44** |
| MIRACL-ko 50k (ko, 213 q) | 0.73 | — | 0.92 | **0.96** |
| Allganize RAG-eval-KO (ko, 1.4k pages, 298 q) | 0.70 | — | 0.69 | 0.72 |
| Mixed ko/ja/zh/en, cross-language questions (19k passages, 200 q) | ≈ 0 | 0.78 | **0.78** | **0.80** |
| **median latency** (warm, M-series) | ~1 ms | ~50 ms | ~50–280 ms | ~2–5 s |

Cross-language questions are English questions about Korean, Japanese or Chinese passages and Korean questions about English ones, mixed in one corpus (MIRACL). The earlier fixed `0.8 · vector + 0.2 · BM25` fusion scored **0.21** there (0.71 with the reranker): BM25 lifted same-language near misses above the right answer. Single-language results above are unchanged by the per-query fusion (within ±0.003). Tier 0 also runs **doc-graph expansion**, which adds ≈ +0.02 on sparse corpora (NFCorpus) and is ~neutral on dense ones (FiQA). Korean BM25 needs the `ko` preprocessor (without it ≈ 0); Korean `Vector` isn't re-measured here (—). Method and data: [research/adr-0002-per-query-fusion.md](research/adr-0002-per-query-fusion.md).

## Versions at a glance

- **≤ 0.15** — core pipeline, daemon, MCP, CJK preprocessors.
- **0.16** — `ir sync` (index + embed in one command); deleted files are removed and moved content reuses cached vectors.
- **0.17** — opt-in research paths: document graph and HNSW ANN index.
- **0.18** — those paths become the default (HNSW ANN, tier-0 graph expansion, reranker window 100); the LLM query expander moves to the calling agent. [ADR-0001](research/adr-0001-default-retrieval-pipeline.md).
- **Next** — per-query hybrid fusion for collections in several languages, parallel tier 1, code domain, rewritten `--help`. [ADR-0002](research/adr-0002-per-query-fusion.md). Existing collections keep working with no re-index.

## Documentation

<details>
<summary><strong>Models</strong></summary>

Models download automatically from HuggingFace Hub on first use (cache: `~/.cache/huggingface/`). `HF_HUB_OFFLINE=1` disables downloads.

| Model | Required for |
|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | vector / hybrid search |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | reranking (optional) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | query expansion (optional) |
| [BGE-M3](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) | Korean-optimized embedding alternative |

**Local models / overrides:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

`IR_*_MODEL` accepts a `.gguf` path, a directory containing a known model, or a HuggingFace repo ID. Search order: env → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → HF Hub. `IR_COMBINED_MODEL` (single model for expand+rerank) is opt-in for experiments only. Switching embedding models requires `ir embed --force`.

**Config directory:**

```bash
export IR_CONFIG_DIR="~/vault/.config/ir"   # portable; supports ~ and $VAR
```

Precedence: `IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir` (deprecated) → `~/.config/ir`.

**GPU:** `IR_GPU_LAYERS=0` forces CPU; `IR_GPU_LAYERS=N` partial offload.

</details>

<details>
<summary><strong>Usage</strong></summary>

**Collections & indexing:**

```bash
ir collection add notes ~/notes
ir collection ls
ir collection rm notes
ir status                    # index health per collection

ir sync [notes] [--force]    # text index + embeddings (the default maintenance command)
ir update [notes] [--force]  # text index only — fast, no models
ir embed [notes] [--force]   # vector repair / re-embedding
```

Indexing is incremental and content-addressed (SHA-256): only changed files are reprocessed, identical content is deduplicated, deleted files are removed, and moved/restored content reuses cached vectors without re-inference.

`ir update` prints the collection path, how many files the scan matched, and how many documents still have no vectors:

```
updating 'notes' (/Users/me/notes)…
  321 scanned: 1 added, 0 updated, 0 deactivated
  1 pending embedding (run `ir embed` or `ir sync`)
```

`0 scanned` means the path or globs matched no files. `ir update --force` rebuilds the text index and drops the collection's stored vectors, so every document is pending until `ir embed` or `ir sync` runs.

If a collection's preprocessor cannot start or stops mid-run, `ir update` warns and indexes the raw text, so every file stays searchable. The next `ir update` with a working preprocessor indexes those files again. Several `ir` processes (CLI runs, MCP servers) can update one collection at the same time. `ir sync` and `ir embed` wait while another process is embedding the same collection.

**Search:**

```bash
ir search "memory safety in rust"                 # hybrid (default)
ir search "sqlite architecture" --mode bm25       # no models
ir search "async patterns" --mode vector
ir search "error handling" -c notes --min-score 0.4

ir search "ownership" --json | --md | --files | --full | --chunk | --quiet
ir search "design" -f "modified_at>=2026-01-01" -f "meta.tags=rust"
ir search "token refresh" --aux                  # also code symbols, after the documents
```

Queries and documents may be in different languages: an English question finds a Korean note in `hybrid` and `vector` mode. `bm25` matches words only.

Filter clauses (`-f`, repeatable, ANDed): fields `path`, `modified_at`, `created_at`, `meta.<name>`; ops `=` `!=` `>` `>=` `<` `<=` `~` `!~`. Dates normalize to UTC RFC3339. Multi-valued frontmatter fields match if **any** element satisfies the clause (including `!=`).

**Retrieve documents:**

```bash
ir get "2026/Daily/2026-04-07.md"              # exact → suffix → substring match
ir get "2026-04-07" -c periodic --section "Log" --max-chars 3000
ir multi-get "a.md" "b.md" --json               # {found, not_found}
```

**Daemon:**

```bash
ir daemon start|stop|status   # auto-starts on first search
```

Warm queries round-trip the Unix socket in ~30ms. On a cold start the first query can return BM25 results immediately while models load in the background.

After you upgrade `ir`, the next search replaces the daemon that the old binary started; that query is answered like a cold start. Homebrew and `cargo install` upgrade while `ir` runs. If you copy the binary by hand, replace the file (`mv` or `install`) instead of writing over it: Linux refuses to write to a running program ("Text file busy"). `--mode bm25` searches never start or contact the daemon. A search stops waiting for a daemon that does not answer within 120 seconds and returns the results it already has.

</details>

<details>
<summary><strong>Korean / Japanese / Chinese preprocessors</strong></summary>

CJK text needs morphological tokenization before BM25 — without it, agglutinated words never match morpheme-level queries (Korean BM25 goes from ~0.00 to useful). The same preprocessor runs at index and query time.

```bash
ir preprocessor install ko    # lindera + ko-dic (official binaries; macOS/Linux)
ir preprocessor install ja    # lindera + ipadic
ir preprocessor install zh    # lindera + jieba
ir preprocessor bind ko wiki  # wire to a collection and re-index
```

Binding `ko` also writes the measured Korean routing default (`fused_strong_product: 0.05`) to that collection; explicit `routing:` config always wins. Per-collection routing overrides (`fused_strong_floor/product`, `bm25_strong_floor/gap`) live in `config.yml` and apply when all searched collections agree.

Any executable can be a preprocessor: UTF-8 lines on stdin → 0-or-1 tokenized lines on stdout, stays alive between lines, passes ASCII-only single words through unchanged. Lindera throughput: ~5,600 Korean docs/s on M-series.

**Why it matters** (MIRACL-Korean):

| preprocessor | BM25 nDCG@10 |
|---|---|
| none | 0.00 |
| lindera (`ko`) | 0.73 (50k-doc sample) |

</details>

<details>
<summary><strong>Code domain — source code as supporting evidence</strong></summary>

Index source files beside a collection's documents. Code files are never search results. Their symbols and doc comments raise, or add, the documents that mention them:

```bash
ir collection add notes ~/project --glob 'docs/**/*.md' --code-glob 'src/**/*.rs' --code-glob 'src/**/*.py'
ir update notes
ir search "drop outdated values after a migration" --json   # finds docs that name the matching function
```

```yaml
collections:
  - name: notes
    path: ~/project
    auxiliary:
      code:
        globs: ["src/**/*.rs", "src/**/*.py"]
        excludes: ["src/generated/**"]   # optional
```

- Supported: `.rs`, `.py`, `.js`/`.jsx`/`.mjs`/`.cjs`, `.ts`/`.tsx`, `.go` (tree-sitter). Functions, methods, types, traits/interfaces, constants and modules are indexed with their doc comments; function bodies are not.
- A document links to a symbol when it names it: `snake_case`, `camelCase` or two-hump `PascalCase` anywhere, or any name inside backticks. It links to a file when it names its path (`src/db/schema.rs` or a unique suffix such as `db/schema.rs`).
- A supported result has an `evidence` field in `--json` output, e.g. `["code:src/cache.rs:invalidate_entry"]`. Without a code domain, results and output are unchanged.
- Removing `auxiliary:` from a collection drops its code index on the next `ir update`.
- `ir status`, MCP `status` and the MCP connect instructions show `aux: code N` only for collections with code. `ir collection add` on a folder with source files and no `--code-glob` prints a one-time hint with the config to add.
- Explicit links, for code a document never names:
  - in the document's frontmatter: `code: [src/keys.rs:rotate_all, src/ledger.rs, parse_config]` (a file path or unique suffix, `path:symbol`, or a symbol name);
  - in a code comment: `// knowledge: design/ledger.md` or `// [[Ledger design]]` (document path or title). It covers the whole file.
- `ir update` warns about each explicit link that matches nothing, for example after a rename, so stale links can be fixed.
- A code match counts only when it covers at least half of the query's words.
- `ir search --aux` (MCP: `aux: ["all"]`; `--aux=code` / `["code"]` to name domains) also returns the matching symbols and files themselves, **after** all documents, with `"kind": "code"`, `path` = source file, `doc_id` = `code:<path>:<line>`, and the signature and doc comment as the snippet. Off by default: code stays auxiliary. Filters (`-f`) apply to documents only.

</details>

<details>
<summary><strong>MCP server — Claude Desktop / Claude Code</strong></summary>

```json
{ "mcpServers": { "ir": { "command": "ir", "args": ["mcp"] } } }
```

Tools: `search` (with `mode`, `limit`, `min_score`, `collections`, `filter`, `aux`), `get`, `multi_get`, `status`, `update`.

`update` is text-only: it refreshes the BM25 index and never computes embeddings. Each result reports `path`, `scanned`, `added`, `updated`, `deactivated`, `pending_embeddings`, and `notes`. Run `ir embed` or `ir sync` in a terminal to embed pending documents. `force: true` drops the collection's existing vectors until then.

`notes` (on `update`) and a second `notes:` block after the `search` results carry the warnings `ir` prints to the terminal, for example a preprocessor that could not start, a collection that is not indexed, or BM25-only results while models load. Both are empty or absent when there is nothing to report.

HTTP mode for remote/multi-client setups:

```bash
ir mcp --http 3620 [--cors '*' | --cors 'https://app.example.com']
```

> HTTP mode is unauthenticated and binds all interfaces — trusted networks only.

</details>

<details>
<summary><strong>Benchmarks & reproduction</strong></summary>

All numbers above are reproducible with the shipped harness:

```bash
scripts/bench.sh nfcorpus            # full per-mode table, cached per git hash
scripts/bench.sh miracl-ko --size 50000 --seed 42
bash scripts/preship.sh              # stability / speed / quality gate on fixtures
```

Runs are resumable (per-query progress survives crashes) and guarded by a memory watchdog on macOS. Historical BEIR results (older pipeline config): reranking added up to +14.5% nDCG@10 over pure vector on ArguAna; fusion alone was not significantly better than pure vector on English corpora — the reranker is where tier-2 value lives.

The HNSW index (usearch) reached 99.2% top-10 overlap with exact search at identical nDCG@10 (MIRACL-ko 50k). The mixed-language set is built from MIRACL; method, data and scripts are described in [ADR-0002](research/adr-0002-per-query-fusion.md).

</details>

<details>
<summary><strong>vs qmd</strong></summary>

ir is a Rust port of [qmd](https://github.com/tobi/qmd) with a different storage model and a persistent daemon.

| | qmd | ir |
|---|---|---|
| Storage | single SQLite | per-collection SQLite (`rm name.sqlite` deletes) |
| Process model | spawn per query | daemon keeps models warm |
| LLM cache | reranker scores | reranker scores + expander outputs |
| Cold / warm query (M4 Max) | 9.5s / 840ms | **3.0s / 30ms** |

</details>

<details>
<summary><strong>Development & schema</strong></summary>

```bash
cargo build [--release]
cargo test                   # no models required
cargo test -- --ignored      # model-dependent tests
```

Per-collection schema: `content` (hash → text), `documents`, `documents_fts` (FTS5), `vectors_vec` (sqlite-vec, cosine), `content_vectors` (chunk metadata), `llm_cache` (reranker scores), `document_metadata` (frontmatter), `meta`, `doc_graph`, `ann_keys`, and the code domain's `artifact_sources`, `artifacts`, `artifact_fts`, `artifact_anchors`, `artifact_edges` (empty without one). Global `expander_cache.sqlite` caches expansion outputs. See [research/pipeline.md](research/pipeline.md) for the staged-async daemon design.

</details>

## Advanced Configuration

Search-pipeline behavior is configured per collection (and globally) in `config.yml` under a `retrieval:` block. Every field is optional — omit it to take the default. Resolution precedence is **`config > env > default`**: a value set here is authoritative and won't be silently overridden by a stray environment variable.

```yaml
# ~/.config/ir/config.yml

# global (daemon): whether to load the in-process LLM query expander
retrieval:
  expander: false            # default — expansion is the calling agent's job

collections:
  - name: notes
    path: ~/notes
    # per-collection pipeline overrides
    retrieval:
      ann: true              # HNSW ANN for vector kNN (exact fallback if unbuilt)
      t0_graph_expand: true  # tier-0 doc-graph seed expansion
      rerank_window: 100     # candidates sent to the reranker
      rerank_keep_window: true
```

| Key | Scope | Default | Meaning |
|---|---|---|---|
| `ann` | collection | `true` | HNSW ANN index for vector search; exact brute-force fallback when absent or stale. |
| `t0_graph_expand` | collection | `true` | BM25 seeds pull `doc_graph` neighbours into the tier-0 candidate list. |
| `rerank_window` | collection | `100` | Number of candidates sent to the tier-2 reranker. |
| `rerank_keep_window` | collection | `true` | Keep judged docs above the un-judged tail. |
| `expander` | global | `false` | Load the in-process LLM query expander; otherwise expansion is delegated to the caller. |

Collections searched together must **agree** on a per-collection value for it to apply; a conflict falls back to the default (same rule as `routing:`). `ir sync` / `ir embed` builds the ANN index and doc graph when the corresponding knob is on (skipped for empty collections). To restore the pre-0.18 pipeline for a collection, set `ann: false`, `t0_graph_expand: false`, `rerank_window: 20`, `rerank_keep_window: false`, and global `expander: true`.

**Alternate config file** — point `ir` at a different `config.yml` without moving the data dir (collections and caches stay put), so you can compare pipeline configurations over one embedded corpus:

```bash
ir --config-path ./variant.yml search "query" -c notes
```

Precedence: `--config-path` > `IR_CONFIG_FILE` > `<config-dir>/config.yml`.

## License

[MIT](LICENSE)
