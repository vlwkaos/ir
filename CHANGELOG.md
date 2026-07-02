## [Unreleased]

### Features

- **Enhanced linked retrieval** (`src/index/units.rs`, `src/db/schema_base.sql`, `src/get.rs`): collection DBs upgrade to schema v3 with indexed content units and explicit unit links. Markdown keeps current chunk behavior; mixed code collections can index best-effort code units with symbols, ranges, language, stored unit text, and deterministic links from `[concept-slug]`, wikilinks, local Markdown links, and frontmatter `related:` / `aliases:`.
- **Mixed collection preset** (`ir collection add --preset mixed`): opt-in globs for Markdown plus mainstream code extensions with common build/vendor excludes. Existing collections and the default Markdown glob are unchanged.
- **Related search output** (`ir search --related N`, MCP `include_related` / `related_limit`): search results can include explicit one-hop related units. Related items are capped at 20 per result, returned separately from primary scores, and are not inferred semantically.
- **Code-unit result metadata**: JSON search results may now include additive optional fields such as `unit_seq`, `unit_kind`, `language`, `symbol`, `start_line`, `end_line`, `indexed_hash`, `indexed_at`, `markers`, and `related`. `chunk_seq` remains for compatibility.
- **Linked synthesis evaluation fixture** (`test-data/fixtures/linked-synthesis`, `scripts/linked-synthesis-eval.py`): model-free evaluation set for mixed code/Markdown synthesis evidence. It fails when search finds a code unit but omits the explicitly linked note evidence needed to answer "why" questions.

### Documentation

- Added `docs/linked-retrieval-agent.md` with a minimal agent search/writing contract for durable knowledge links.

## [0.15.0] - 2026-05-06

### Features

- **Linux GPU backends** (`Cargo.toml`, `src/llm/mod.rs`): added `llama-cuda`, `llama-rocm`, and `llama-vulkan` Cargo features. Build with `--features llama-cuda` (NVIDIA), `llama-rocm` (AMD), or `llama-vulkan` (cross-platform) to enable GPU acceleration on Linux. `IR_GPU_LAYERS` now defaults to 99 when any GPU backend feature is compiled in (was macOS-only). Runtime backend is detected via `ggml_backend_dev_*` enumeration and reported in daemon startup log (e.g. `loading models (CUDA)...`).

- **Nix flake** (`flake.nix`): added `flake.nix` with packages for all four platforms. Linux default package uses OpenMP/CPU; named outputs `#cuda`, `#rocm`, `#vulkan` enable the respective GPU backends with correct `buildInputs`. macOS package links Metal/Foundation/Accelerate frameworks automatically. Thanks to @bglgwyngag for the initial implementation ([#15](https://github.com/vlwkaos/ir/pull/15)).

## [0.14.1] - 2026-05-04

### Features

- **`--cors` flag for `ir mcp --http`** (`src/mcp.rs`, `src/cli/mod.rs`): adds `Access-Control-Allow-Origin` support for browser-hosted MCP clients. `--cors '*'` allows any origin and disables rmcp's DNS-rebinding host check; `--cors 'https://...'` sets an exact-match origin. Omitting `--cors` leaves existing behavior unchanged. Exposes `mcp-session-id` header and allows `last-event-id` on SSE reconnect.

### Chores

- **hf-hub upgraded to 0.5.0** (`Cargo.toml`): switched from `hf-hub 0.3` (native-tls / OpenSSL) to `hf-hub 0.5.0` with `default-features = false, features = ["ureq"]`. Removes OpenSSL and native-tls from the dependency tree; TLS handled by ureq 3's rustls default. The sync `api::sync::ApiBuilder` path is unchanged.

## [0.14.0] - 2026-04-30

### Fixes

- **Portable SQLite extension ABI** (`src/db/mod.rs`): sqlite-vec auto-extension init function now uses `*mut *mut c_char` (from `std::os::raw`) instead of `*mut *mut i8` for the error message pointer, matching the stable C ABI. Previous code was technically UB on targets where `c_char` is unsigned.
- **Preprocessor arg expansion** (`src/preprocess.rs`): `PreprocessHandle::spawn` now applies `expand_path` to all command args, not just the binary path. Fixes `$IR_DIR/preprocessors/jieba` being passed literally to lindera — caused zh (and any post-v0.12.0 fresh ko/ja install) indexing to fail with "dictionary path does not exist".

### Dev / Benchmark Tooling

- **GitHub Actions CI/Release**: added `.github/workflows/ci.yml` (build + test on push/PR) and `.github/workflows/release.yml` (cross-platform binary build, GitHub release, Homebrew tap update on tag push). `TAP_TOKEN` secret required in repo settings for the Homebrew step.

- **Chinese (zh) benchmark infrastructure**: `scripts/download-miracl-zh.sh` downloads the MIRACL-ZH corpus (~132K passages) from HuggingFace; `scripts/bench.sh miracl-zh` runs the benchmark; `scripts/generate-fixtures.sh` generates `test-data/fixtures/miracl-zh-mini/` (2000-doc sample, seed=42); placeholder `expected.json` with `_uncalibrated` flags added. Calibrate with `scripts/calibrate-fixtures.sh miracl-zh-mini` after download.
- **Chinese synthetic fixture** (`test-data/fixtures/synthetic-zh/`): 20-doc Chinese corpus with BM25 fingerprint terms (玄武/朱雀/青龙/白虎/勾陈), semantic synonym pairs, distractors, and edge cases including punctuation-only lines. Exercises zh preprocessor pipeline end-to-end via `scripts/preship.sh --fixture synthetic-zh`. Calibrate with `scripts/calibrate-fixtures.sh synthetic-zh` after `ir preprocessor install zh`.
- **zh integration test** (`src/preprocess.rs`): `#[ignore]` test `zh_preprocessor_tokenizes_chinese` verifies jieba segmentation, ASCII pass-through, and punctuation handling. Run with `cargo test -- --ignored zh_preprocessor` after `ir preprocessor install zh`.
- **README.zh.md**: Full Simplified Chinese translation of README.md.
- **Benchmark isolation preprocessor symlink** (`scripts/bench-env.sh`): `bench_env_init` now symlinks the live `preprocessors/` directory into each isolated bench state dir. Previously, bench scripts using `$IR_DIR/preprocessors/...` references would fail in isolated state; only old absolute-path configs worked by accident.

## [0.13.0] - 2026-04-23

### Breaking

- **Combined-model default removal** (`src/daemon.rs`, `src/llm/combined.rs`): dedicated expander + reranker is now the only default tier-2 path. A local Qwen combined GGUF is no longer auto-activated from model search dirs; combined mode is opt-in via `IR_COMBINED_MODEL` only and is intended for explicit testing or experiments.

### Features

- **Per-collection routing overrides** (`config.yml`, `src/types.rs`, `src/search/hybrid.rs`): collections can now carry optional BM25/fused strong-signal threshold overrides under a `routing:` block. This gives runtime threshold tuning a stable config surface without changing the global defaults. Overrides apply only when all searched collections agree; mixed searches with conflicting values fall back to the global defaults.

  ```yaml
  collections:
    - name: wiki-ko
      path: ~/wiki
      preprocessor: [ko]
      routing:
        fused_strong_product: 0.05
  ```

  Fields: `fused_strong_floor`, `fused_strong_product`, `bm25_strong_floor`, `bm25_strong_gap`.

- **Built-in Korean bind default** (`src/main.rs`): binding the built-in `ko` preprocessor alias now also writes the current Korean fused routing default (`routing.fused_strong_product: 0.05`) for that collection when no explicit routing override exists yet.
  - Existing `ko`-bound collections are **not** auto-migrated. Add the `routing:` block manually to `config.yml` or unbind/rebind `ko` to apply.
  - Rationale: on the sampled Korean holdout `miracl-ko-s50000-p42`, fused `0.05` outperforms `0.06` on both quality and latency (`nDCG@10=0.9650`, `R@10=0.9813`, `med=431.5ms` vs `nDCG@10=0.9603`, `R@10=0.9766`, `med=440.4ms`).

### Improvements

- **Cold-search daemon warmup** (`src/main.rs`): `ir search` now kicks off daemon startup immediately, but a cold first query no longer waits for model download/load if BM25 already found usable results. The daemon continues warming in the background so follow-up queries land on the hotter path.
- **Large-corpus embed stall fix** (`src/index/embed.rs`): `ir embed` no longer loads every pending document body into memory before starting work. Pending docs are now counted once and streamed in small batches by content hash, which makes progress visible immediately and avoids the full-corpus RAM spike that could make large MIRACL-Ko resumes look hung for an hour before the first progress update.
- **Indexer progress bar** now shows docs/sec and counts from the hash phase (previously only showed apply phase progress).

### Dev / Benchmark Tooling

- **Pre-ship regression gate** (`scripts/preship.sh`): three-axis regression check (stability, speed, performance) across test fixtures before release. Catches hang/crash/timeout (stability), throughput and latency regressions (speed), and nDCG/Recall regressions (performance). Portable: works with macOS bash 3.2, uses perl alarm fallback when GNU timeout is unavailable.
- **Committed test fixtures** (`test-data/fixtures/synthetic-en/`): 20-doc English corpus with discriminative BM25 fingerprint terms, semantic synonym pairs, distractors, and edge cases. Calibrated expected.json with 10% buffer floors. Catches pipeline regressions in ~30s without any model download.
- **Korean canary fixture** (`test-data/fixtures/miracl-ko-mini/expected.json`): placeholder for issue #13-class deadlock detection. Populate with `scripts/generate-fixtures.sh` after downloading miracl-ko.
- **Fixture calibration** (`scripts/calibrate-fixtures.sh`): measures actual metrics per mode, writes calibrated baselines to expected.json with 10% buffer floors. Per-mode `_uncalibrated` flag prevents uncalibrated modes from failing the gate.
- **Pool-size variance study** (`scripts/pool-size-study.sh`, `scripts/pool-size-aggregate.py`): sweeps miracl-ko at multiple corpus sizes × multiple seeds to find the smallest pool with stable nDCG stddev < 0.005. Writes `research/pool-size-study.md`. Current minimum stable pool: 10000 docs; active research default: 50000 docs because 10000 often saturates ranking metrics. Pools at or below the 503 mandatory qrel-linked docs are treated as deterministic floors, not variance evidence.
- **Progress reporting**: timestamped `_log()` in bench.sh and signal-sweep.sh; stall detector in signal-sweep.sh (STALL DETECTED at 120s no-output with issue context); tqdm progress bars in beir-eval.py (materialize, run, signals, sample).
- **Benchmark summary output** (`scripts/bench.sh`): prints one row per available mode (`bm25`, `vector`, `hybrid`) instead of collapsing a run to a single best-mode row. Makes baseline capture usable directly from the benchmark command.
- **Benchmark resume** (`scripts/bench.sh`, `scripts/beir-eval.py run`): rerunning the same dataset after a crash now reuses a prepared collection when it is already ready for the requested mode, instead of redoing the entire prepare stage. Query scoring also resumes from per-query sidecar progress when `--output` is set, so a crash mid-run does not force the query loop to restart from zero. Cache validation distinguishes `bm25`-only results from full `all`-mode results.
- **Benchmark pipeline pinning** (`scripts/bench.sh`, `scripts/bench-env.sh`): non-BM25 benchmark runs now force the dedicated expander + reranker path and restart the benchmark daemon before scoring. This prevents local combined-model auto-detect or stale daemon state from silently changing the benchmark pipeline. Benchmark state now also exports `IR_CONFIG_DIR`, eliminating the deprecated `XDG_CONFIG_HOME` warning during benchmark runs.
- **Sampled benchmarks** (`scripts/bench.sh --size N --seed N`): large BEIR corpora can now be benchmarked directly on a sampled pool without hand-running `beir-eval.py sample` first. Sampled runs get distinct dataset labels, collections, and result-cache directories such as `miracl-ko-s10000-p42`, so they do not overwrite full-corpus baselines.
- **Research harness** (`scripts/research-harness.sh`, `scripts/signal-sweep.sh --sample-only`, `scripts/threshold-validate.py`, `research/experiment.md`): the benchmark workflow is now consolidated around a maintainer entrypoint with `baseline`, `signals`, `thresholds`, and `validate-thresholds` subcommands. Korean threshold research now defaults to sampled `miracl-ko --size 50000` pools for better metric headroom, while `10000` remains the fast stable floor.
- **Benchmark safety watchdog** (`scripts/bench.sh`, `scripts/bench-env.sh`): macOS benchmark runs now keep Metal enabled but watch system free memory, swapouts, and runaway `ir` CPU usage during long `prepare` / `run` phases. The wrapper aborts unsafe runs before they drag the machine into swap-heavy or CPU-fallback territory. Thresholds are tunable via `IR_BENCH_MIN_FREE_PCT`, `IR_BENCH_MAX_IR_CPU_PCT`, `IR_BENCH_CPU_STRIKES`, or can be disabled with `IR_BENCH_GUARD=0`.
- **Threshold override env vars** (`src/search/hybrid.rs`): strong-signal and BM25 shortcut thresholds can be overridden via env vars during research runs. This keeps the shipped defaults unchanged while the harness validates candidate thresholds against a holdout collection.
- **Tier-2 router dataset export** (`scripts/export-tier2-router-data.py`, `research/experiment.md`): signal sweeps can now be converted directly into smoltrain JSONL for a tiny `skip_tier2` vs `run_tier2` classifier trained on real benchmark behavior instead of hand labels.
- **Tier-2 router smoltrain prep** (`scripts/prepare-tier2-router-smoltrain.py`, `research/experiment.md`): exported router datasets can now be turned into a self-contained `smoltrain` workspace with `train.jsonl`, `train_balanced.jsonl`, `eval.jsonl`, `taxonomy.yaml`, and `world.json`.
- **Router bundle wrapper** (`scripts/router-data.sh`): router bundle prep now has a separate entrypoint for Korean-only, FiQA-only, or mixed training data, so router research stays isolated from the working `scripts/research-harness.sh` baseline / threshold flow.
- **Tier-1 router benchmark path** (`src/search/hybrid.rs`, `scripts/beir-eval.py`, `scripts/router-bench.py`): research runs can force `hybrid` to return tier-1 fused results only, collect `tier1.jsonl` alongside normal signal files, and benchmark a trained router offline against a holdout without changing the shipped runtime path.
- **Research conclusions documented** (`research/experiment.md`, READMEs, benchmark skill): current maintainer guidance explicitly prefers stricter threshold gating before any runtime router integration. FiQA stays on the current fused threshold, Korean threshold research uses sampled `miracl-ko --size 50000`, and router work remains offline until it clearly beats simple gating on holdout.

## [0.12.0] - 2026-04-20

### New Features

- `IR_CONFIG_DIR` env var: override the config/data directory with `~` and `$VAR` expansion.
  Safe to use in MCP configs (`.mcp.json`) synced across machines with different usernames.
  Precedence: `IR_CONFIG_DIR` > `XDG_CONFIG_HOME/ir` > `~/.config/ir`.

### Deprecations

- `XDG_CONFIG_HOME` is deprecated as the config dir override for ir. It still works but
  prints a warning. Use `IR_CONFIG_DIR` instead.

### Improvements

- All path env vars (`IR_*_MODEL`, `IR_MODEL_DIRS`, `IR_CONFIG_DIR`) now support `~` and
  `$VAR`/`${VAR}` expansion. Previously, `~` in these vars was treated literally and silently
  failed to resolve.
- Collection paths in `config.yml` now support `~` and `$VAR` notation. Portable paths are
  preserved on write; expansion happens at load time.
- Preprocessor commands installed via `ir preprocessor install` are now stored as
  `$IR_DIR/preprocessors/...` instead of absolute paths, making them portable across machines.
  Existing absolute-path commands continue to work. Re-run `ir preprocessor install <lang>`
  to migrate to the portable format.
- When BM25 returns no results (semantic query with no keyword overlap), the daemon wait
  timeout is extended from 3s to 10s — the daemon is the only source of results in this
  case, so a slow cold start no longer silently returns nothing. Diagnostic hints are printed
  at all daemon fallback sites (start failure, timeout, query error) to guide follow-up
  (`ir embed <collection>`, model path check).

## [0.11.2] - 2026-04-18

### Bug Fixes

- Preprocessor install: pin lindera download to v3.0.5 instead of resolving `/releases/latest`
  at install time. Prevents silent breakage if lindera ships a major version with changed CLI
  flags or tokenizer output format. ([`2cdbd78`](https://github.com/vlwkaos/ir/commit/2cdbd78))

## [0.11.1] - 2026-04-18

### Bug Fixes

- Preprocessor pipe deadlock on large Korean collections ([#13](https://github.com/vlwkaos/ir/issues/13)): `process_line` now uses
  a sentinel-based protocol to handle lines where the official lindera CLI produces no output
  (e.g. punctuation-only lines where all tokens are filtered by `--token-filter`). Previously,
  `read_line()` would block forever waiting for output that never arrived, deadlocking both
  processes at 0% CPU. Manifested as a hang at ~60k docs when indexing MIRACL-Ko or other large
  Korean corpora. ([`2084e4e`](https://github.com/vlwkaos/ir/commit/2084e4e0b62fcace710834150eeba8cf7d338a59))

## [0.11.0] - 2026-04-17

### Bug Fixes

- BM25 now uses OR semantics for natural-language queries (>3 terms): stop words are
  stripped and remaining terms are ORed. Short keyword queries (≤3 non-stop terms) keep
  AND semantics. Fixes near-zero recall on question-format queries (e.g. `ir search --mode bm25
  "what are the symptoms of diabetes"` previously returned almost nothing due to AND forcing
  all stop words to match).
- Preprocessor subprocess: `process_line` now skips empty/whitespace-only lines instead of
  sending them to the subprocess. Fixes a deadlock when indexing markdown with blank lines
  using official lindera CLI (which emits no output for empty input when `--token-filter` is active).

### Breaking

- **`ir preprocessor install ko/ja/zh` now uses official lindera releases.** Previously `ir`
  bundled its own preprocessor binaries in GitHub release artifacts (unreliable — artifacts
  were occasionally missing). Starting v0.11.0, `ir preprocessor install` downloads the
  official lindera CLI binary and per-language dictionary directly from lindera's GitHub
  releases. Chinese (`zh`) switches from a custom bigram tokenizer to lindera + jieba.

  **Migration required if you used a preprocessor before v0.11.0:**
  ```bash
  ir preprocessor install ko   # (or ja / zh)
  ir update <collection> --force
  ```
  Existing config entries pointing to old bundled binaries are stale. Search silently degrades
  (BM25 without tokenization) until reinstalled.

## [0.10.0] - 2026-04-17

### Features

- `-f/--filter "FIELD OP VALUE"` on `ir search`: general structured filter supporting built-in fields (`path`, `modified_at`, `created_at`) and YAML frontmatter fields (`meta.<name>`). Operators: `=`, `!=`, `>`, `>=`, `<`, `<=`, `~` (contains), `!~` (not-contains). Multiple `-f` flags are ANDed. Date values are normalized to UTC RFC3339. Applied at all three search pipeline tiers so each exit point returns correctly filtered results.
- MCP `search` tool gains a structured `filter` array (`[{field, op, value}]`) with full JSON schema — LLM clients see typed enum choices for operators.
- Frontmatter metadata indexed into a new `document_metadata` table at `ir update` time; supports all scalar values, tag arrays (one row per element), and nested keys.

### Bug Fixes

- Daemon tier-2: reranker without expander now correctly reranks tier-1 fused results (reranking is useful without expansion; expansion alone is harmful, -0.53% nDCG on NFCorpus).
- Daemon tier-2: `IR_COMBINED_MODEL` load failure now falls back to dedicated models with an explicit warning instead of silently disabling tier-2.
- Daemon tier-2: conflict between `IR_COMBINED_MODEL` and dedicated model env vars now warns before loading (combined wins).
- Daemon tier-2: `QMD_EXPANDER_MODEL` / `QMD_RERANKER_MODEL` legacy aliases now correctly trigger dedicated mode instead of falling through to auto-detect.

### Breaking

- `--modified-after` / `--modified-before` CLI flags removed (were unreleased). Use `-f "modified_at>=YYYY-MM-DD"` and `-f "modified_at<=YYYY-MM-DD"`.
- MCP `SearchInput.modified_after` / `modified_before` fields removed. Use `filter: [{field: "modified_at", op: ">=", value: "YYYY-MM-DD"}]`.
- Collection DBs upgrade to schema version 2 on first use. A one-time backfill populates `document_metadata` from existing frontmatter (sub-second for <10k docs). No manual migration required.

## [0.9.0] - 2026-04-15

### Features

- `ir search --quiet` / `-q`: suppress all stderr output (progress indicators, daemon log lines). Useful for scripting. Conflicts with `--verbose`. ([`486b360`](https://github.com/vlwkaos/ir/commit/486b360))
- `IR_COMBINED_MODEL`: new canonical env var for the unified Qwen3.5 GGUF (replaces both expander + reranker roles). `IR_QWEN_MODEL` still accepted but emits a deprecation warning on load. ([`6524540`](https://github.com/vlwkaos/ir/commit/6524540))
- `IR_*_MODEL` env vars now accept HuggingFace repo IDs (`owner/name`) in addition to local file and directory paths. Setting e.g. `IR_EMBEDDING_MODEL=ggml-org/bge-m3-Q8_0-GGUF` downloads and caches the model automatically on first use. ([`9325335`](https://github.com/vlwkaos/ir/commit/9325335))
- BGE-M3 added to the auto-download registry (`ggml-org/bge-m3-Q8_0-GGUF`). Download progress shown in foreground terminal; daemon loads from cache instantly.
- Download UX improved: structured message before HF progress bar shows model name, size hint, source URL, cache location, and offline tip.
- Download errors now include actionable fixes: retry, `HF_HUB_OFFLINE=1`, manual download URL, cache path to clear on corruption.

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
