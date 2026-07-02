# Linked Retrieval Agent Guide

Use this when an agent should search a mixed Markdown/code collection and preserve durable knowledge links.

## Setup

```bash
ir collection add project . --preset mixed
ir update project
ir embed project
```

`--preset mixed` is opt-in. Existing Markdown collections keep their old behavior.

## Search

```bash
ir search "why does hybrid search skip reranking" \
  --collection project \
  --chunk \
  --related 3 \
  --json
```

Read primary results first. Treat `related` as explicit linked context, not a second ranking list. If a result has no related items, fall back to normal search or `ir get`.

## Evaluation Fixture

The repository includes a model-free synthesis evidence check:

```bash
cargo build --bin ir
python3 scripts/linked-synthesis-eval.py --ir-bin target/debug/ir
```

The fixture lives at `test-data/fixtures/linked-synthesis/`. It verifies that mixed code/Markdown search returns the code unit plus explicitly linked notes containing the evidence needed to answer synthesis-style "why" questions. This checks context assembly, not LLM prose generation.

## Writing Durable Knowledge

Use one stable marker for each durable concept:

```md
---
related:
  - src/search/hybrid.rs#is_bm25_strong_signal
---

# Strong Signal Shortcut

The shortcut avoids model work when lexical evidence is decisive.

See [strong-signal-shortcut].
```

In code, put markers in comments or doc comments near the relevant symbol:

```rust
// [strong-signal-shortcut]
// Keep this threshold conservative; false positives skip reranking.
fn is_bm25_strong_signal(...) -> bool { ... }
```

Supported explicit links:

- `[concept-slug]`
- `[[Note]]`, `[[Note#Heading]]`, `[[path/to/file]]`, `[[Note|Alias]]`
- local Markdown links such as `[label](docs/design.md#heading)`
- frontmatter `related:` and `aliases:`

## Staleness Rule

Line ranges are not durable identity. Agents should trust the indexed unit text returned by `--chunk` and treat live file navigation as best-effort until `ir update` refreshes `indexed_hash`, symbol, and range metadata.

## Minimal Prompt

```text
Before broad grep, run `ir search "<query>" --chunk --related 3 --json`.
Use primary results for relevance and related results for explicit context.
When adding durable knowledge, add one `[concept-slug]` marker and, when useful, one wikilink or local markdown link.
Do not invent links to files outside configured collections; unresolved links are acceptable.
```
