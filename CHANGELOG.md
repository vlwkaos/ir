## [Unreleased]

### Features

- `ir search --quiet` / `-q`: suppress all stderr output (progress indicators, daemon log lines). Useful for scripting. Conflicts with `--verbose`.
- `IR_COMBINED_MODEL`: new canonical env var for the unified Qwen3.5 GGUF (replaces both expander + reranker). `IR_QWEN_MODEL` still accepted but emits a deprecation warning on load.
- `IR_*_MODEL` env vars now accept HuggingFace repo IDs (`owner/name`) in addition to local file and directory paths. Setting e.g. `IR_EMBEDDING_MODEL=ggml-org/bge-m3-Q8_0-GGUF` downloads and caches the model automatically on first use.
- BGE-M3 added to the auto-download registry (`ggml-org/bge-m3-Q8_0-GGUF`). Download progress is shown in the foreground terminal; the daemon loads from cache and starts instantly.
- Download UX improved: a structured message is printed before the HF progress bar showing model name, size hint, source URL, cache location, and a tip for offline use.
- Download errors now include actionable fixes: network retry, offline mode (`HF_HUB_OFFLINE=1`), manual download URL, and cache directory path for clearing corrupt downloads.

### Breaking

- Unrecognized `IR_*_MODEL` values (not a file, directory, or known HF repo ID) now error immediately instead of silently falling through to the default model. Users with leftover garbage env vars will see an error with an "Accepted forms" list. Unset the env var to restore default behavior.

## [0.8.2] - 2026-04-15

### Features

- `ir get --section "Heading"`: extract a named section from a document by ATX heading text (case-insensitive, CommonMark-compliant, code-fence-aware). Section runs from the matched heading to the next heading of the same or higher level. MCP `get` tool gains an equivalent `section` parameter. Returns empty string when heading not found. ([`dfc7d2d`](https://github.com/vlwkaos/ir/commit/dfc7d2dd5a898e636b90fe600c6425a22eecaa5f))

### Bug Fixes

- Chunker: replace post-loop tail merge with inline rebalance-or-absorb. When a normal split would leave a sub-minimum tail, the chunk boundary is pulled back so both the current chunk and the tail meet `MIN_CHUNK_SIZE_TOKENS` (100 tokens / 400 chars). Fixes a latent infinite loop where a semantic break point inside the overlap window caused `start` to stop advancing, producing an unbounded `Vec<Chunk>` and OOM. ([`ddced38`](https://github.com/vlwkaos/ir/commit/ddced38b493f0d59cd392b3374b837728a392a47))

## [0.8.1] - 2026-04-14

### Features

- `ir get` / `ir multi-get`: `--max-chars` truncates output to N characters; `--offset` skips the first N characters (`get` only). Both use char-safe slicing (correct for CJK and other multibyte content). MCP `get` and `multi_get` tools gain equivalent `max_chars` and `offset` parameters. Solves MCP tool-result overflow for large documents ([`9e54c69`](https://github.com/vlwkaos/ir/commit/9e54c69c986d34cb1c91e1cf704e77950e7fe713))
- `ir search --chunk`: populates result content with the best-matching chunk text from vector/hybrid search. MCP `search` gains equivalent `include_chunk` parameter. Results from BM25-only queries leave content empty (no chunk identity for FTS results) ([`9e54c69`](https://github.com/vlwkaos/ir/commit/9e54c69c986d34cb1c91e1cf704e77950e7fe713))
- MCP `search`: `full` parameter now exposed (was CLI-only). Set `full=true` to include full document text inline in search results ([`9e54c69`](https://github.com/vlwkaos/ir/commit/9e54c69c986d34cb1c91e1cf704e77950e7fe713))
- `SearchResult.chunk_seq`: best-matching chunk index propagated through the full pipeline (vector kNN → score fusion → reranking → daemon IPC → CLI/MCP). Appears in JSON output for vector/hybrid results; absent for BM25-only results ([`9e54c69`](https://github.com/vlwkaos/ir/commit/9e54c69c986d34cb1c91e1cf704e77950e7fe713))

### Breaking

- `ir search --json` and MCP `search` JSON output now includes `"chunk_seq": N` for vector/hybrid results. BM25-only results are unaffected (field omitted). Strict schema validators must be updated

## [0.8.0] - 2026-04-14

### Features

- `ir get` and `ir multi-get` CLI commands: retrieve full document text by path from any registered collection. Accepts collection-relative paths (`2026/file.md`), vault-root paths (`Notes/2026/file.md`), suffix matches, and substring matches — same matching rules as the MCP tools. `ir get` exits 1 on miss; `ir multi-get` reports all misses to stderr and exits 1 if any path was not found ([`8c30d84`](https://github.com/vlwkaos/ir/commit/8c30d849139371832c6402dba63d7313cc804fab))
- Vault-root path resolution: when the first component of a path matches the directory name of a collection's root (e.g. `Notes` for a collection at `~/Notes`), `ir` now strips the prefix and resolves against the matching collection before falling back to a global search. Fixes silent not-found for paths like `0. PeriodicNotes/2026/Daily/2026-04-07.md` ([`8c30d84`](https://github.com/vlwkaos/ir/commit/8c30d849139371832c6402dba63d7313cc804fab))

### Bug Fixes

- LIKE injection: `%` and `_` in file paths were treated as SQLite wildcards in suffix/substring lookup tiers, causing false-positive matches. All user-supplied paths are now escaped with an `ESCAPE '\\'` clause before binding. Fix applies to both CLI and MCP ([`8c30d84`](https://github.com/vlwkaos/ir/commit/8c30d849139371832c6402dba63d7313cc804fab))

### Docs

- README: added "Retrieve documents" section with `ir get` / `ir multi-get` examples including vault-root paths and output format flags ([`9c0cb4a`](https://github.com/vlwkaos/ir/commit/9c0cb4a6355352eddd5aa351ccf66e81d3aae260))

## [0.7.2] - 2026-04-13

### Features

- `ir mcp`: added `get` and `multi_get` tools for full document retrieval by path — enables Claude Desktop and claude.ai to read document content without a separate filesystem MCP server. `get(path)` resolves via exact → suffix → substring match; `multi_get(paths[])` batches multiple lookups and returns found documents with a list of unmatched paths ([`5541b79`](https://github.com/vlwkaos/ir/commit/5541b79fc1f12ee6344d2ce68a60354bbeb76cda))

## [0.7.1] - 2026-04-08

### Bug Fixes

- `ir preprocessor install` now works on all supported platforms: macOS universal binary (`darwin-universal`) is fetched for both Apple Silicon and Intel; Linux x86\_64 and aarch64 are also supported. Previously the CLI constructed `darwin-arm64` / `darwin-x86\_64` URLs that never existed in the release assets, causing a 404 for every user ([`src/main.rs`](https://github.com/vlwkaos/ir/blob/main/src/main.rs))
- Corrupt or failed tar extraction during preprocessor install now returns a clear error and cleans up the downloaded tarball instead of silently continuing with a missing binary

### Docs

- Preprocessor install section now lists supported platforms: macOS (universal, Intel + Apple Silicon) and Linux (x86\_64, aarch64)

### Build

- `scripts/release-extras.sh`: Linux preprocessor cross-compilation added via `cross` (x86\_64-unknown-linux-musl, aarch64-unknown-linux-musl); skips gracefully with install hint if `cross` is not available

## [0.7.0] - 2026-04-07

### Features

- `ir mcp` subcommand: native Model Context Protocol server for Claude Desktop and Claude Code integration; exposes `search`, `status`, and `update` tools; supports stdio (default) and HTTP (`--http <port>`) transports ([`f76c98c`](https://github.com/vlwkaos/ir/commit/f76c98c7a3a7c990e3fec7dd4c22f3b33510a042))

## [0.6.7] - 2026-03-30

### Bug Fixes

- Switching embedding models (e.g. EmbeddingGemma 768d to BGE-M3 1024d) no longer silently corrupts vectors; `ir embed --force` auto-detects dimension mismatch and rebuilds the vector table ([`3d7d211`](https://github.com/vlwkaos/ir/commit/3d7d211))

### Docs

- Korean embedding model guide: BGE-M3 setup, comparison table, KURE-v1 experimental note, expander degradation warning ([`9cfa6ee`](https://github.com/vlwkaos/ir/commit/9cfa6ee))
- Universal binary builds enforced in release scripts ([`5576911`](https://github.com/vlwkaos/ir/commit/5576911))

## [0.6.6] - 2026-03-24

### Features

- `ir search --full` now prints stored document text inline; falls back to snippet if content is unavailable ([`ace970a`](https://github.com/vlwkaos/ir/commit/ace970a))

## [0.6.5] - 2026-03-24

### Features

- Japanese preprocessor (`ir preprocessor install ja`) now uses the self-contained `lindera-tokenize-ja` binary (Lindera + ipadic) instead of a MeCab shell script ([`1d89779`](https://github.com/vlwkaos/ir/commit/1d89779))
- Japanese POS filter now includes フィラー (filler words), matching Lucene kuromoji default stoptags ([`1d89779`](https://github.com/vlwkaos/ir/commit/1d89779))

### Docs

- Added Japanese and Chinese preprocessor research to `research/experiment.md` — ipadic vs unidic comparison, Mode::Decompose penalty rationale, bigram vs word segmentation literature with MIRACL benchmarks ([`4b4b161`](https://github.com/vlwkaos/ir/commit/4b4b161))
- README: installation, quick start, and Korean preprocessor docs ([`0256e86`](https://github.com/vlwkaos/ir/commit/0256e86), [`4974fed`](https://github.com/vlwkaos/ir/commit/4974fed), [`2494ef7`](https://github.com/vlwkaos/ir/commit/2494ef7))

## [0.6.4] - 2026-03-20

### Features

- `ir collection ls` now shows bound preprocessors per collection as `[alias, ...]` ([`644ee1f`](https://github.com/vlwkaos/ir/commit/644ee1f))

### Bug Fixes

- `ir preprocessor bind`: re-index failure is now a warning instead of a fatal error — binding is saved regardless ([`644ee1f`](https://github.com/vlwkaos/ir/commit/644ee1f))
- `ir preprocessor bind`: clearer error when alias is not registered, with `install` or `add` hint depending on whether alias is a known bundled preprocessor ([`644ee1f`](https://github.com/vlwkaos/ir/commit/644ee1f))
- Daemon race: parallel `ir search` invocations no longer spawn duplicate daemons; flock on `daemon.lock` ensures only one client spawns the process ([`d45d817`](https://github.com/vlwkaos/ir/commit/d45d817))

## [0.6.3] - 2026-03-19

### Features

- `ir preprocessor bind <alias> [collection]` — wire a preprocessor to a collection and re-index; omitting collection shows an interactive multiselect picker ([`6ed9515`](https://github.com/vlwkaos/ir/commit/6ed9515))
- `ir preprocessor unbind <alias> <collection>` — remove a preprocessor from a collection and re-index ([`6ed9515`](https://github.com/vlwkaos/ir/commit/6ed9515))
- `ir preprocessor install` now launches the collection picker after download ([`6ed9515`](https://github.com/vlwkaos/ir/commit/6ed9515))
- `ir preprocessor list` shows install hint in section header and custom add tip ([`4230d82`](https://github.com/vlwkaos/ir/commit/4230d82))

### Docs

- Preprocessor guide updated for binary download install, correct lang codes, and new bind/unbind commands ([`5e134bf`](https://github.com/vlwkaos/ir/commit/5e134bf))

## [0.6.2] - 2026-03-19

### Features

- `ir preprocessor list` now shows registered preprocessors and available bundled ones ([`9193ad9`](https://github.com/vlwkaos/ir/commit/9193ad9))
- `ir preprocessor remove` is soft by default; `--delete` / `-d` also removes the binary if installed under the ir preprocessors dir ([`9193ad9`](https://github.com/vlwkaos/ir/commit/9193ad9))
- `ir preprocessor install` now downloads pre-built binaries from GitHub releases instead of building from crates.io ([`9193ad9`](https://github.com/vlwkaos/ir/commit/9193ad9))
- Preprocessor binaries (`lindera-tokenize`, `lindera-tokenize-ja`, `bigram-tokenize-zh`) are now built and uploaded as release assets ([`9193ad9`](https://github.com/vlwkaos/ir/commit/9193ad9))

## [0.6.0] - 2026-03-19

### Features

- Japanese and Chinese preprocessing support (WIP) ([`b281bc6`](https://github.com/vlwkaos/ir/commit/b281bc6))
- Korean preprocessor switched to Lindera tokenizer with compound noun decompounding ([`66f6808`](https://github.com/vlwkaos/ir/commit/66f6808))
- Preprocessor pipeline now warms subprocess per-request for lower latency ([`e316fb4`](https://github.com/vlwkaos/ir/commit/e316fb4))
- Daemon hot-reloads config on SIGHUP ([`b25e3ea`](https://github.com/vlwkaos/ir/commit/b25e3ea))
- Preprocessor subcommand added to CLI ([`d330d47`](https://github.com/vlwkaos/ir/commit/d330d47))
- Preprocessor parallelism support ([`854784e`](https://github.com/vlwkaos/ir/commit/854784e))
- DB threads preprocessor commands ([`9112e64`](https://github.com/vlwkaos/ir/commit/9112e64))
- Compound noun decompounding benchmark added to eval ([`fa210e8`](https://github.com/vlwkaos/ir/commit/fa210e8))

### Bug Fixes

- BM25 strong-signal items now propagate correctly through the pipeline ([`64a5fd6`](https://github.com/vlwkaos/ir/commit/64a5fd6))
- Consolidated preprocessor install list and fixed download URL ([`555be35`](https://github.com/vlwkaos/ir/commit/555be35))

### Refactor

- Korean preprocessing consolidated to Lindera ([`355625e`](https://github.com/vlwkaos/ir/commit/355625e))

## [0.5.1] - 2026-03-12

### Features

- Tiered async daemon startup: BM25 runs in-process immediately; embedder binds socket (tier 1, ~1s); expander+reranker load in background (tier 2, ~3–5s) ([`101da40`](https://github.com/vlwkaos/ir/commit/101da40))

### Bug Fixes

- `--mode bm25` now returns directly without a daemon round-trip; `--mode vector` no longer short-circuits on a strong BM25 signal ([`b86279c`](https://github.com/vlwkaos/ir/commit/b86279c))
- Tier-0 strong-signal threshold raised: raw BM25 floor 0.75 / gap 0.10 (was using fused thresholds calibrated on 0.80·vec+0.20·bm25) ([`b86279c`](https://github.com/vlwkaos/ir/commit/b86279c))
- Guard `start_in_background` with `is_running()` check to prevent orphaning a live daemon ([`b86279c`](https://github.com/vlwkaos/ir/commit/b86279c))
- Skip tier-2 signal file when both expander and reranker fail to load ([`b86279c`](https://github.com/vlwkaos/ir/commit/b86279c))
- Use `open_rw` for client-side collection DBs: no schema init, fails fast on missing DB ([`b86279c`](https://github.com/vlwkaos/ir/commit/b86279c))

## [0.5.0] - 2026-03-10

### Features

- `ir collection set-path`: update the root path of an existing collection without re-creating it ([`1ee1330`](https://github.com/vlwkaos/ir/commit/1ee1330))
- Eval: paired t-test across pipeline modes; per-dataset label in summary table ([`d7db622`](https://github.com/vlwkaos/ir/commit/d7db622))

### Refactor

- Progressive search pipeline: BM25+vector always fused; expansion gated on expander presence; strong-signal shortcut exits before expansion ([`2e987f0`](https://github.com/vlwkaos/ir/commit/2e987f0))

### Benchmark (4 BEIR datasets, nDCG@10)

EmbeddingGemma 300M + qmd-expander-1.7B + Qwen3-Reranker-0.6B:

| Dataset | BM25 | Vector | Hybrid | +Reranker |
|---|---|---|---|---|
| NFCorpus (323q) | 0.2046 | 0.3898 | 0.3954 | **0.4001** |
| SciFact (300q) | 0.0500 | 0.7847 | 0.7873 | **0.7797** |
| FiQA (648q) | 0.0298 | 0.4324 | 0.4266 | **0.4567** |
| ArguAna (1406q) | 0.0012 | 0.4264 | 0.4263 | **0.4879** |

BM25 fusion provides no statistically significant lift on any dataset (paired t-test). Reranker adds up to +14.5% on conversational/argument retrieval.

### Other

- Crate published as `ir-search` on crates.io (`ir` name was taken) ([`39dfefd`](https://github.com/vlwkaos/ir/commit/39dfefd))
- BEIR dataset download script (`scripts/download-beir.sh`) ([`ff71879`](https://github.com/vlwkaos/ir/commit/ff71879))

## [0.4.0] - 2026-03-06

### Features

- Daemon mode: keeps models warm between queries, auto-starts on first search ([`a9d1996`](https://github.com/vlwkaos/ir/commit/a9d1996))
- Global expander output cache (`~/.config/ir/expander_cache.sqlite`): repeated queries skip LLM inference entirely ([`407ea55`](https://github.com/vlwkaos/ir/commit/407ea55))
- Reranker score cache now includes model_id in key, preventing cross-model cache collisions ([`8b6c82e`](https://github.com/vlwkaos/ir/commit/8b6c82e))
- Eval harness: cache query embeddings and per-query results across runs ([`fe51038`](https://github.com/vlwkaos/ir/commit/fe51038))

### Performance (macOS M4 Max, vs qmd, same models and query)

| | ir | qmd | ratio |
|---|---:|---:|---|
| Cold (no cache) | 3.0s | 9.5s | 3× faster |
| Warm (daemon + caches hot) | 30ms | 840ms | 28× faster |

Cold: expander ~2.9s + reranker inference (ir caps at 20 candidates, qmd at 40).
Warm: both expander and reranker cache-hit; qmd pays ~800ms process spawn per call.

### Refactor

- Extract `scoring.rs` (Scorer trait + batch inference) and `generate.rs` (autoregressive generation) from reranker ([`8b6c82e`](https://github.com/vlwkaos/ir/commit/8b6c82e))

### Other

- DSPy experiment: BootstrapFewShot + ChatAdapter; Qwen3.5 underperforms trio by ~1.9% nDCG@10 ([`6f268fc`](https://github.com/vlwkaos/ir/commit/6f268fc))
- bench.sh script for sequential multi-config eval runs ([`55b61f5`](https://github.com/vlwkaos/ir/commit/55b61f5))

## [0.3.1-pre] - 2026-03-03

### Bug Fixes

- Auto CPU fallback when Metal context creation fails in sandboxed environments ([`299780a`](https://github.com/vlwkaos/ir/commit/299780a))
- Open search connections immutable/read-only to avoid WAL shm writes in sandbox ([`aa307c2`](https://github.com/vlwkaos/ir/commit/aa307c2))

### Other

- Move config and data to XDG-style `~/.config/ir` (cross-platform, sandbox-accessible) ([`295b3bb`](https://github.com/vlwkaos/ir/commit/295b3bb))
- DSPy optimizer: add structured logging, `--resume` flag, ollama smoke-test ([`2fdd0e9`](https://github.com/vlwkaos/ir/commit/2fdd0e9))
