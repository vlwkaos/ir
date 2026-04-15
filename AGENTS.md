# ir — Agent Instructions

## Crate

Package name on crates.io is `ir-search` (name `ir` was taken).
Binary names remain `ir` and `eval`. See @Cargo.toml.

## Commands

```bash
cargo build                        # dev build (ir + eval binaries)
cargo build --release --bin ir     # release build
cargo test                         # unit tests (fast, no models needed)
cargo test -- --ignored            # includes LLM tests (require model files)
```

Benchmark runner (requires BEIR dataset):
```bash
scripts/bench.sh --data test-data/nfcorpus baseline "B:IR_COMBINED_MODEL=~/local-models/Qwen3.5-0.8B-Q8_0.gguf"
```
Logs go to `logs/` (gitignored).

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `IR_EMBEDDING_MODEL` | auto-detect | Path to embedding GGUF |
| `IR_EXPANDER_MODEL` | auto-detect | Path to expander GGUF (qmd-1.7B) |
| `IR_RERANKER_MODEL` | auto-detect | Path to reranker GGUF (Qwen3-0.6B) |
| `IR_COMBINED_MODEL` | unset | Unified Qwen3.5 GGUF — replaces both expander + reranker (`IR_QWEN_MODEL` deprecated alias) |
| `IR_GPU_LAYERS` | `99` on macOS | Number of layers offloaded to GPU |
| `IR_FORCE_CPU_BACKEND` | unset | Set to `1` to disable Metal |
| `IR_LLAMA_LOGS` | unset | Set to `1` to enable llama.cpp verbose logging |
| `IR_MODEL_DIRS` | `~/local-models/` | Colon-separated extra model search dirs |
| `XDG_CONFIG_HOME` | `~/.config` | Overrides config/data dir base |

Model search order: `IR_*_MODEL` env → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/`

`QMD_EMBEDDING_MODEL`, `QMD_EXPANDER_MODEL`, `QMD_RERANKER_MODEL` are also checked as fallbacks.

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

- **Crate name mismatch**: package is `ir-search`, but all internal `use` statements in `src/bin/eval.rs` reference `ir_search::` (snake_case). Don't change these back to `ir::`.
- **LLM tests are `#[ignore]`**: `cargo test` skips them. Run `cargo test -- --ignored` only when model files are present.
- **sqlite-vec must be registered before any connection opens**: `ensure_sqlite_vec()` uses `sqlite3_auto_extension` (process-global). Called once via `OnceLock` in `db/mod.rs`.
- **`LlamaBackend` is a singleton**: `OnceLock<LlamaBackend>` in `src/llm/mod.rs`. Loading a second model in the same process does NOT call `init()` again — this is intentional.
- **Daemon requires restart after binary change**: `ir search` auto-starts the daemon but won't restart a running one. Kill it with `ir daemon stop` after rebuilding.
- **`ir embed` prints "GPU context unavailable, falling back to CPU"** in sandboxed environments — normal, not an error.
- **Never run embedding or LLM inference in background shell tasks** — sandboxed shells have no Metal access, so they fall back to CPU and peg it. Hand these off to the user's terminal instead.
- **Strong-signal shortcut**: raw BM25 top ≥ 0.75 AND gap ≥ 0.10 skips all LLM work (`is_bm25_strong_signal`); fused top*gap ≥ 0.06 AND top ≥ 0.40 skips tier-2 (`is_strong_signal`). Both in `src/search/hybrid.rs`.

## Release

release.flow: rust

```bash
# Homebrew + crates.io
~/.claude/skills/rust-release/release.sh "$VERSION" "ir" "vlwkaos/ir" "$HOME/ws-ps/homebrew-tap"
cargo publish   # publishes as ir-search
```

Requires `dangerouslyDisableSandbox: true` — gh CLI reads `~/.config/gh` (sandbox read deny list); tap writes to `~/ws-ps/homebrew-tap` (add to sandbox write allowlist to avoid this).

## good-to-go

Recurring audit axes (auto-maintained by /good-to-go):
- README.md + README.ko.md must both be updated for any user-facing feature (CLI flags, env vars, output formats)
- CHANGELOG.md Unreleased section must cover: new CLI flags, env var renames/deprecations, breaking behavior changes
- Enum variants in types.rs must be wired to a CLI flag or MCP field — check with `rg 'Variant::' src/ | grep -v test`
- Preprocessor protocol tests must use line-flushing commands (cat, rev) not buffered ones (tr, sed, sort)
- IR_COMBINED_MODEL is the canonical combined-model env var; IR_QWEN_MODEL is a deprecated alias — do not promote the alias in new docs
