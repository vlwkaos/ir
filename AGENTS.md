# ir — Agent Instructions

## Crate

Package name on crates.io is `ir-search` (name `ir` was taken).
Binary name is `ir`. See @Cargo.toml.

## Setup

```bash
git config core.hooksPath .githooks   # activate pre-commit clippy hook (once per clone)
```

## Commands

```bash
cargo build                        # dev build
cargo build --release --bin ir     # release build
cargo test                         # unit tests (fast, no models needed)
cargo test -- --ignored            # includes LLM tests (require model files)
```

Benchmark runner (drives the real `ir` binary; requires BEIR dataset):
```bash
scripts/bench.sh fiqa              # bench current HEAD on FiQA
scripts/bench.sh fiqa v0.9.0       # compare HEAD vs v0.9.0
scripts/bench.sh miracl-ko         # Korean MIRACL benchmark
```
Results cached at `logs/results/{dataset}/{git7}.json` (gitignored).

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `IR_EMBEDDING_MODEL` | auto-detect | Path to embedding GGUF |
| `IR_EXPANDER_MODEL` | auto-detect | Path to expander GGUF (qmd-1.7B) |
| `IR_RERANKER_MODEL` | auto-detect | Path to reranker GGUF (Qwen3-0.6B) |
| `IR_COMBINED_MODEL` | unset | Unified Qwen3.5 GGUF — replaces both expander + reranker (`IR_QWEN_MODEL` deprecated alias) |
| `IR_GPU_LAYERS` | `99` when GPU backend compiled in, else `0` | Number of layers offloaded to GPU |
| `IR_FORCE_CPU_BACKEND` | unset | Set to `1` to force CPU (disables Metal/CUDA/ROCm/Vulkan) |
| `IR_LLAMA_LOGS` | unset | Set to `1` to enable llama.cpp verbose logging |
| `IR_MODEL_DIRS` | `~/local-models/` | Colon-separated extra model search dirs |
| `IR_CONFIG_DIR` | `~/.config/ir` | Override config/data dir. Supports `~` and `$VAR` expansion. |
| `IR_CONFIG_FILE` | unset | Override the `config.yml` file path only (data dir unchanged). Set by the `--config-path` CLI arg and inherited by the daemon. Precedence: `--config-path` > `IR_CONFIG_FILE` > `<config-dir>/config.yml`. |
| `XDG_CONFIG_HOME` | `~/.config` | **Deprecated** — use `IR_CONFIG_DIR` instead. Still works but emits a warning. |
| `IR_BENCH_SIGNALS` | unset | Research: emit `SIGNAL_FUSED\ttop\tgap` to pipeline log for threshold tuning |
| `IR_DISABLE_SHORTCUTS` | unset | Research: disable BM25 + fused strong-signal shortcuts for A/B benchmarking |
| `IR_FORCE_TIER1_ONLY` | unset | Research: force hybrid to return tier-1 fused results only (skip tier-2) |
| `IR_STRONG_SIGNAL_FLOOR_OVERRIDE` | unset | Research: override fused strong-signal floor threshold |
| `IR_STRONG_SIGNAL_PRODUCT_OVERRIDE` | unset | Research: override fused strong-signal product threshold |
| `IR_STRONG_SIGNAL_PRODUCT_PREPROCESSED_OVERRIDE` | unset | Research: override fused strong-signal product for preprocessed (Korean) collections |
| `IR_BM25_STRONG_FLOOR_OVERRIDE` | unset | Research: override BM25 strong-signal floor threshold |
| `IR_BM25_STRONG_GAP_OVERRIDE` | unset | Research: override BM25 strong-signal gap threshold |
| `IR_ALLOW_EXPANSION_WITHOUT_SCORER` | unset | Research: allow expansion without reranker (harmful in production: -0.53% nDCG on NFCorpus) |
| `IR_GRAPH_BUILD` | unset | Research: build `doc_graph` cosine-kNN edges during `ir embed` |
| `IR_GRAPH_K` | `10` | Research: top-k neighbors per doc at graph build |
| `IR_GRAPH_T0_EXPAND` | on (0.18) | Tier-0 graph expansion via capped score propagation. **0.18 default; prefer `retrieval.t0_graph_expand` in config.yml — this env var is a deprecated override.** |
| `IR_GRAPH_T0_MODE` | `cap` | Research: `rrf` switches tier-0 expansion to RRF fusion (measured harmful; kept for A/B) |
| `IR_GRAPH_T1_CONSENSUS` | unset | Research: tier-1 neighborhood consensus boost (measured neutral + fire-rate harmful; kept for A/B) |
| `IR_GRAPH_T2_EXPAND` | unset | Research: GAR-style rerank pool expansion at tier 2; injected docs bypass metadata filters |
| `IR_GRAPH_DECAY` | `0.8` | Research: activation decay γ for graph propagation |
| `IR_GRAPH_SEEDS` | `10` | Research: number of top results used as propagation seeds |
| `IR_GRAPH_LAMBDA` | `0.2` | Research: consensus blend weight (T1 consensus) |
| `IR_GRAPH_RRF_WEIGHT` | `0.5` | Research: graph-mass weight in RRF mode |
| `IR_GRAPH_T1_EXPAND` | unset | Research: tier-1 cap injection into fused list (measured no-op — cosine neighbors already in pool) |
| `IR_GRAPH_T1_INJECT` | `30` | Research: max docs injected by T1 expand |
| `IR_GRAPH_T2_INJECT` | `30` | Research: max docs injected by T2 expand (saturates at 30 — inject60 = 323/323 ties) |
| `IR_GRAPH_AS_EXPANDER` | unset | Research: skip LLM expander at tier 2; graph injection + reranker only (quality ≈ rerank-only) |
| `IR_RERANK_WINDOW_OVERRIDE` | `100` (0.18) | Tier-2 rerank window size. **0.18 default 100; prefer `retrieval.rerank_window` — deprecated override.** |
| `IR_RERANK_KEEP_WINDOW` | on (0.18) | Judged rerank window always outranks un-judged tail. **0.18 default on; prefer `retrieval.rerank_keep_window` — deprecated override.** |
| `IR_ANN` | on (0.18) | `hnsw` enables the usearch ANN sidecar for vector kNN (exact fallback when stale/absent). **0.18 default on; prefer `retrieval.ann` — deprecated override.** |
| `IR_DISABLE_EXPANDER` | expander off (0.18) | Legacy inverted flag. **0.18 drops the in-process expander by default; prefer global `retrieval.expander` in config.yml — deprecated override.** |
| `IR_ANN_M` | `16` | Research: HNSW connectivity (usearch `connectivity`) |
| `IR_ANN_EF_CONSTRUCTION` | `200` | Research: HNSW build-time expansion |
| `IR_ANN_EF` | `200` | Research: HNSW search-time expansion (recall knob; 99.2% top-10 overlap, nDCG@10 = exact at default on 50k) |
| `IR_BENCH_MAX_SWAPOUT_DELTA` | `65536` | Research: bench watchdog tolerance (pages, ~1 GiB) for benign system swapout drift before abort; `free_pct` floor still guards real exhaustion. Set `0` for strictest pristine-latency runs (bench-env.sh) |

Config dir precedence: `IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir` (deprecated) → `~/.config/ir`

Retrieval profile (0.18): the search-pipeline knobs (`ann`, `t0_graph_expand`, `rerank_window`, `rerank_keep_window`, `expander`) resolve through one `RetrievalProfile` (`src/search/profile.rs`) with precedence **`config > env > default`**. `config.yml` `retrieval:` (per-collection + top-level `expander`) is the authoritative home; the `IR_*` env vars above are a deprecated convenience layer (removed in a later release). Active default = `DEFAULT_V018` (ANN/t0/window-100/keep on, expander off). Fine-tuning knobs (`IR_ANN_EF`, `IR_GRAPH_DECAY`, …) stay env-only.

Model search order: `IR_*_MODEL` env → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/`

`QMD_EMBEDDING_MODEL`, `QMD_EXPANDER_MODEL`, `QMD_RERANKER_MODEL` are also checked as fallbacks.

All path env vars (`IR_CONFIG_DIR`, `IR_MODEL_DIRS`, `IR_*_MODEL`) support `~` and `$VAR`/`${VAR}` expansion.

Note: `IR_DIR` is set internally at startup (= resolved `ir_dir()` value). It appears in preprocessor commands stored in `config.yml` as `$IR_DIR/preprocessors/...` so they are portable. Not user-facing.

## Data Paths

- Config: `~/.config/ir/config.yml`
- Collection DBs: `~/.config/ir/collections/{name}.sqlite`
- Expander cache: `~/.config/ir/expander_cache.sqlite`
- Daemon socket: `~/.config/ir/daemon.sock`

## Architecture

### Search Pipeline

Three-tier escalation. Each tier runs only if the previous tier's result isn't strong enough.

| Tier | Models | Enables |
|------|--------|---------|
| 0 | none | BM25 (FTS5), in-process |
| 1 | Embedder | Vector, hybrid score-fusion (0.80·vec + 0.20·bm25) |
| 2 | Expander + Scorer | Query expansion (lex/vec/hyde → RRF) + reranking |

Strong-signal shortcut: raw BM25 top ≥ 0.75 AND gap ≥ 0.10 → skip Tier 1+2 entirely (`src/search/hybrid.rs:is_bm25_strong_signal`). Expander without scorer is a no-op (`hybrid.rs:112`).

See @research/pipeline.md for diagrams.

### Daemon Startup

Staged async: BM25 runs in-process immediately. Daemon starts in background.

- Tier 1 ready: embedder loaded → socket bound → client unblocks (waits up to 3s)
- Tier 2 ready: expander + reranker loaded → tier2 signal file written → client re-queries if needed (waits up to 7s)

Idle timeout: 3600s (configurable via `ir daemon start --timeout`).

## Known Gotchas

- **LLM tests are `#[ignore]`**: `cargo test` skips them. Run `cargo test -- --ignored` only when model files are present.
- **sqlite-vec must be registered before any connection opens**: `ensure_sqlite_vec()` uses `sqlite3_auto_extension` (process-global). Called once via `OnceLock` in `db/mod.rs`.
- **`LlamaBackend` is a singleton**: `OnceLock<LlamaBackend>` in `src/llm/mod.rs`. Loading a second model in the same process does NOT call `init()` again — this is intentional.
- **Daemon requires restart after binary change**: `ir search` auto-starts the daemon but won't restart a running one. Kill it with `ir daemon stop` after rebuilding.
- **`ir embed` prints "GPU context unavailable, falling back to CPU"** in sandboxed environments — normal, not an error.
- **Never run embedding or LLM inference in background shell tasks** — sandboxed shells have no Metal access, so they fall back to CPU and peg it. Hand these off to the user's terminal instead.
- **Strong-signal shortcut**: raw BM25 top ≥ 0.75 AND gap ≥ 0.10 skips all LLM work (`is_bm25_strong_signal`); fused top*gap ≥ 0.06 AND top ≥ 0.40 skips tier-2 (`is_strong_signal`). Both in `src/search/hybrid.rs`.

## Release

release.flow: rust-ci

- Release atomic references resolve under `~/.claude/skills/release/atomic/`; run release state checks separately because an empty `rg --files knowledge/sessions` exits 1
- Before editing release guidance, inspect the checked-in section with `rg -n -A45 -B3 '^## Release|release\.flow|GitHub Actions|git push origin' AGENTS.md`; session-provided instructions may omit local additions
- For release secret scans, run double-quoted and single-quoted assignment patterns as separate `rg` commands; combining both quote classes in one zsh command can produce an unmatched-quote error
- Resolve changelog commit URLs with `git rev-parse HEAD`; never manually expand an abbreviated hash
- If `git push origin main` is rejected as non-fast-forward, do not push the release tag. Fetch and inspect `origin/main`, rebase clean unpushed release commits, rerun affected checks, update rebased changelog hashes, and recreate the local tag before retrying

`.github/workflows/release.yml` is the release source of truth. A pushed `v*` tag runs checks, builds release artifacts, creates the GitHub release, updates the Homebrew tap, and publishes `ir-search` to crates.io. Do not also run `cargo publish`, `gh release create`, or update the tap manually.

```bash
# Bump version
sed -i '' 's/^version = ".*"/version = "'"$VERSION"'"/' Cargo.toml
cargo check --quiet  # updates Cargo.lock

# Finalize CHANGELOG (CI awk extracts release notes by version heading)
sed -i '' "s/^## \[Unreleased\]/## [$VERSION] - $(date +%Y-%m-%d)/" CHANGELOG.md

# Commit + tag + push (triggers CI release workflow)
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "v$VERSION"
git tag -a "v$VERSION" -m "v$VERSION"
git push origin main
git push origin "refs/tags/v$VERSION"
```

Prerequisites: `TAP_TOKEN` and `CRATES_IO_TOKEN` secrets set in GitHub repo settings.
CI handles: build, GitHub release, Homebrew tap update, and crates.io publish.

## good-to-go

- README.md + README.ko.md + README.zh.md must all be updated for any user-facing feature (CLI flags, env vars, output formats)
- CHANGELOG.md Unreleased section must cover: new CLI flags, env var renames/deprecations, breaking behavior changes
- Enum variants in types.rs must be wired to a CLI flag or MCP field — check with `rg 'Variant::' src/ | grep -v test`
- Preprocessor protocol tests must use `cat` only — `rev` uses full stdio buffering in pipe mode on macOS and deadlocks. `tr`, `sed`, `sort` also buffer.
- IR_COMBINED_MODEL is the canonical combined-model env var; IR_QWEN_MODEL is a deprecated alias — do not promote the alias in new docs
- src/search/filter.rs must have unit tests for eval_clause + match_op — these are pure functions with no DB dependency; easy to test, and zero coverage is a gap [resolved v0.10.0: 9 tests added]
- FilterOp::Ne on multi-valued fields uses any-match semantics (same as all ops): `meta.tags!=rust` passes if ANY tag != "rust"; document this in README filter table, not just code comments [resolved v0.10.0: documented in both READMEs and tested]
- items_after_test_module: in Rust files, keep non-test items (impl fns, helper fns) BEFORE any #[cfg(test)] mod block — clippy::items_after_test_module will fail the build
- build_query_natural in db/fts.rs is used for all production BM25 queries; uses OR + stop word stripping for natural-language queries, AND for short keyword queries
- cargo clippy --all-targets -- -D warnings must pass before release; check llm/ files for needless_borrow when updating llama.cpp bindings
- Format touched Rust files with `rustfmt --edition 2024 --config skip_children=true <files>`; plain `rustfmt` on crate roots recursively reformats untouched child modules
- `cargo test` does not guarantee `target/debug/ir` is refreshed; run `cargo build --bin ir` before CLI smoke tests
- warn_stale_preprocessor() in src/main.rs is a migration shim for ≤0.9.x users — removed at v0.13.0
- Research-only env vars (IR_BENCH_SIGNALS, IR_DISABLE_SHORTCUTS, IR_FORCE_TIER1_ONLY, IR_STRONG_SIGNAL_*_OVERRIDE, IR_BM25_STRONG_*_OVERRIDE, IR_ALLOW_EXPANSION_WITHOUT_SCORER) must NOT appear in README; CHANGELOG may name them only under "Dev / Benchmark Tooling"; document in CLAUDE.md env table
- preprocess.rs sentinel protocol (IRSENTINEL): process_line() sends content line + IRSENTINEL, reads until IRSENTINEL — prevents pipe deadlock when lindera emits no stdout for all-filtered lines (e.g. punctuation-only). Custom preprocessors must pass ASCII-only single-word lines through unchanged. When any preprocessor command changes (new binary, flags, or external tool replacing custom code): run probe `printf '.\n안녕하세요\ntest\n' | <new_command> 2>/dev/null | wc -l` — must equal 3, or WARN and confirm sentinel covers the 0-output case. Test suite must include at least one test where process_line() is called with a line the subprocess drops.
- IR_DIR is set internally at startup (= resolved ir_dir() value); appears in preprocessor commands as $IR_DIR/preprocessors/... for portability — do not expose in user-facing docs
- All path env vars (IR_CONFIG_DIR, IR_MODEL_DIRS, IR_*_MODEL) support ~ and $VAR expansion via expand_path() in src/config/mod.rs — tests for this must use ENV_LOCK mutex to prevent parallel env var interference
- scripts/preship.sh must pass (exit 0 or 2) before any signal-sweep run or release; run `--bm25-only` for fast CI gate, full for pre-release
- Run `cargo build --release --bin ir` immediately before `bash scripts/preship.sh`; preship only builds when `target/release/ir` is absent and otherwise reuses the existing binary
- Default pool size for MIRACL-Ko signal sweeps: 50000 docs. Minimum stable floor from the variance study: 10000 docs. Do not use pool sizes <= 503 for between-seed variance decisions; those pools collapse to the mandatory qrel-linked docs and are deterministic.
- zh fixture (test-data/fixtures/synthetic-zh) must be calibrated before shipping zh-related changes; run `ir preprocessor install zh && scripts/calibrate-fixtures.sh synthetic-zh` then commit updated expected.json
- scripts/preship.sh --fixture synthetic-zh must pass (exit 0 or 2) before any zh-related release
- zh sentinel probe before any zh preprocessor command change: `printf '。\n你好世界\ntest\n' | <zh-cmd> 2>/dev/null | wc -l` — must equal 3
- PreprocessHandle::spawn expands all args via expand_path (not just the binary); preprocessor commands may use $IR_DIR or ~ in any arg position — IR_DIR is set by ir at startup (main.rs:54), tests that spawn preprocessors directly must set IR_DIR manually
- speed floors in expected.json must be calibrated from a full run (with embed) when the fixture includes vector/hybrid modes; BM25-only calibration sets unrealistically high floors that fail on embed runs — reset min_index_docs_per_s to 5 and max_query_p50_ms to 2000 until full GPU calibration runs
- bench_env_init preprocessors symlink: state dirs get a symlink to the live source preprocessors on each init, replacing stale real directories; after ir preprocessor install, bench scopes that ran before the install have stale config.yml — delete scope's config.yml to force re-copy on next run

- Uncertain about project term/schema/convention/prior decision → `/seek <topic>` first (lightweight KB lookup; same tier as grep/Glob).
