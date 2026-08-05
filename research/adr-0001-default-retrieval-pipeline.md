# ADR-0001 — Default retrieval pipeline for v0.18

- **Status:** Accepted; shipped as the **default in 0.18.0** (ANN, tier-0 graph
  expansion, wide rerank window + keep-window on; LLM expander off). The
  O(N·log N) graph-from-ANN build is deferred to a 0.18.x follow-up — 0.18.0
  builds the doc graph via the existing exact pass.
- **Date:** 2026-08-01
- **Supersedes default behavior of:** 0.17.x (all features below were env-gated / off).

## Context

The 0.17 research line added, all off by default, a document-similarity graph
(`doc_graph`), tiered graph-expansion variants, an HNSW ANN index (`usearch`),
and rerank-window controls. We benchmarked them on public corpora
(NFCorpus, FiQA, MIRACL-ko 50k sample, Allganize RAG-eval-KO). The load-bearing
measurements (paired per-query t-tests):

| Change | Result |
|---|---|
| Tier-0 graph cap injection | NFCorpus **+0.024 nDCG@10** (t=+4.9, 36W/0L), +23% R@1000; zero-harm by construction; inert on saturated corpora |
| Rerank window 100 + keep-window, **no LLM expander** | FiQA **+0.041 over fusion** (t=+6.6), MIRACL-ko +0.046, Allganize-KO +0.033 — the 0.6B reranker captures the full tier-2 lift |
| LLM expander contribution once window is wide | ≈ 0 on every tested corpus (all monolingual, query↔doc vocabulary overlap) |
| HNSW ANN (`ef=200`) | 99.2% top-10 overlap vs exact, **nDCG@10 identical to exact**, tier-1 median 198ms→162ms |
| Graph build via blocked matmul | O(N²); 44s / 61k chunks — still superlinear |

Two structural facts drove the design:

1. **HNSW's base layer is itself an approximate kNN graph.** Once the ANN index
   exists, `doc_graph` can be built by per-doc self-query against it —
   **O(N·log N)** — instead of the O(N²) brute-force/matmul pass. ANN and the
   graph stop being two indexes and become one substrate plus a cheap derivation.
2. **The LLM expander's job (query rewriting / HyDE) is something the calling
   agent already does.** ir is a search primitive; an agent driving it can expand
   queries itself. Dropping the built-in 1.7B expander removes a model from the
   default footprint and ~3.5s from tier-2 latency, and the wide-window reranker
   recovers the quality on every corpus we could measure.

## Decision

In 0.18, make the following the **default** pipeline (each remains overridable):

- **ANN (HNSW) on by default** for vector kNN — sublinear query search and the
  substrate the graph builds from. Exact brute-force remains the fallback whenever
  the index is absent or stale, so correctness never depends on the index existing.
- **`doc_graph` built from the ANN index** (per-doc self-query), not brute force.
- **Tier-0 graph expansion on** — a zero-harm win on sparse-result corpora.
- **Wide rerank window (100) + keep-window on** — the quality change that makes
  the no-expander pipeline viable.
- **LLM query expander off by default** — expansion is delegated to the caller.
  The expander remains available opt-in for callers that want in-process expansion.

0.17.x keeps all of the above **opt-in** (env-gated), plus the tier-2 metadata
filter-bypass fix. 0.18 flips the defaults; no schema change is required —
`doc_graph` and `ann_keys` already exist (created empty since 0.17).

### Configuration and precedence

The scattered per-call `env_flag`/`env::var` reads (~10 sites across the
index-build, daemon, and query layers) are replaced by **one resolved
`RetrievalProfile`** (`src/search/profile.rs`), resolved once per owning layer:
`resolve_for_query` (per-collection, agreement rule reused from `RoutingConfig`),
`resolve_for_daemon` (global — `expander` is a process-level model-load decision),
`resolve_for_build` (per-collection at `ir embed`). All defaults live in one
`DEFAULT_V017` / `DEFAULT_V018` const.

Three inputs, resolved **`config > env > default`** — identical to how
`RoutingConfig` already resolves, so the whole binary shares one precedence rule:

- **config.yml** — the authoritative, persistent home. A `retrieval:` block,
  hand-edited, mirroring the existing `routing:` UX (per-collection + a top-level
  block for the global `expander` knob). `Option`/serde-default: omit a knob to
  get the built-in default. A value set here **cannot be silently masked** by a
  stale environment variable.
- **env** — a **deprecated convenience layer**. The existing `IR_*` presence knobs
  keep working for quick one-off sweeps, but only fill in where config is silent;
  they are documented as deprecated and removed progressively in later versions
  once benchmarks and users have moved to config-path. The daemon prints a startup
  note whenever a research override is active (so "why am I not seeing 0.18
  behavior?" is one glance at the log). Fine-tuning calibration knobs
  (`IR_ANN_EF`, `IR_GRAPH_DECAY`, …) are **not** part of the profile and stay env.
- **built-in default** — `DEFAULT_V018`. No config, no env → the 0.18 pipeline.

`--config-path` (file-level) overrides **only the config.yml file, not the data
dir** (collections, caches, daemon socket stay put). This decouples "which config"
from "which data" so a benchmark can vary pipeline config across candidates while
reusing one embedded corpus. Precedence: `--config-path` arg > `IR_CONFIG_DIR` >
default. Benchmarks graduate from env sweeps to per-candidate config files
(reproducible: run = a checked-in file, not remembered shell exports).

Documentation lives under **README → Advanced Configuration** (EN/KO/ZH) — between
"invisible research env var" and "front-page feature": default users get
`DEFAULT_V018` with zero config; power users editing pipeline behavior go to
Advanced. Genuinely per-query knobs (`--mode`) stay CLI args, outside the profile.

### Index time (0.18)

```
INDEX TIME  (ir sync / ir embed)

  documents ──► embed ──► vectors (sqlite-vec)
              │
              ▼
  ┌────────────────────────┐
  │  BUILD  HNSW / ANN     │   usearch · incremental · exact fallback
  │         ~80s / 50k     │
  └───────────┬────────────┘
              │   doc_graph built FROM the ANN index:
              │   per-doc self-query, O(N·log N) — not O(N²)
              ▼
  ┌────────────────────────┐
  │  BUILD  doc_graph      │   kNN neighbour edges · cosine weight
  └───────────┬────────────┘
              ▼
           ◎ collection ready
```

### Query time (0.18 defaults)

```
QUERY TIME  (0.18 defaults — graph + ANN ON, LLM expander OFF)

  ir search "query"
              │
              ▼
  ┌────────────────────────┐
  │  PREPROCESS  ko/ja/zh  │   lindera · index+query · optional
  └───────────┬────────────┘
              ▼
  ┌────────────────────────┐
  │  TIER 0  BM25          │   FTS5 · in-process
  │      + graph expand    │   seed neighbours from doc_graph (NEW)
  └───────────┬────────────┘
              │   strong signal ?   ── yes ──►  ◎ results
              │  no
              ▼
  ┌────────────────────────┐
  │  TIER 1  Hybrid        │   vector via ANN (NEW) + bm25 fusion
  └───────────┬────────────┘
              │   strong signal ?   ── yes ──►  ◎ results
              │  no
              ▼
  ┌────────────────────────┐
  │  TIER 2  Rerank        │   + Qwen3-Reranker-0.6B
  │     window 100         │   + graph pool · keep-window (NEW)
  │     NO LLM expander    │   query expansion → caller/agent (NEW)
  └───────────┬────────────┘
              ▼
           ◎ results
```

## Migration (seamless)

No manual step for existing collections:

1. Upgrade binary. Schema tables already present (0.17), so nothing to migrate.
2. Until the next `ir sync`, the ANN index and graph are absent → queries fall
   back to exact search and no graph expansion — **behaviorally identical to 0.17**.
3. Next `ir sync`/`ir embed` builds the ANN index and derives `doc_graph` from it
   (one-time cost, then incremental). Search transparently upgrades. No downtime.

Staged so each stage ships green:

- **Stage 1** — introduce the resolved `RetrievalProfile` as the one presence
  source of truth (replaces scattered per-call `env_flag` reads), with
  `config > env > default` precedence and the `retrieval:` config block; land the
  file-level `--config-path` override (decoupled from the data dir) so benchmarks
  can validate Stages 2–3 via per-candidate config files. Defaults are
  `DEFAULT_V017` → behavior-preserving, existing tests pass unchanged.
- **Stage 2** — build `doc_graph` from the ANN index with a brute-force fallback.
  Gate: T0 graph nDCG on NFCorpus within tolerance of the brute-force-built graph
  (ANN-derived edges are approximate, so this is an empirical gate, not bit-parity).
- **Stage 3** — flip the profile defaults (ANN/T0/window/keep-window on, expander
  off). Gate: the full public-corpus benchmark matches or beats 0.17 defaults.

## Consequences

- **Footprint down**: the 1.7B expander no longer loads by default (less memory,
  faster tier-2 readiness). **Up**: the ANN index roughly doubles on-disk vector
  storage (~3KB/vector).
- **Approximate by default**: vector recall is 99.2% (nDCG identical to exact on
  the validation set), with exact fallback whenever the index isn't ready.
- **Callers own query expansion.** Agents that relied on in-process expansion must
  expand queries themselves or opt the expander back in. Documented in the 0.18
  release notes.
- **config.yml becomes the authoritative override; env is deprecated.** Presence
  knobs move to a `retrieval:` config block (`config > env > default`). The `IR_*`
  presence env vars keep working as a deprecated convenience layer and are removed
  progressively in later versions; the daemon logs a note when any override is
  active. Fine-tuning calibration env vars are unaffected.
- **Untested regime**: all measurements are monolingual with query↔doc vocabulary
  overlap. The expander's theoretical value is cross-lingual / vocabulary-mismatch
  retrieval, which none of the public corpora exercise. If that regime matters,
  expansion belongs in the agent (which can translate/HyDE) rather than a built-in default.

## Rejected alternatives

- **Ship the flips in 0.17** — rejected. 0.17 is stabilized as a bug-free, opt-in
  line; a default behavior change belongs in a minor bump (0.18) with its own notes.
- **Keep scattered per-feature env flags, just invert their defaults** — rejected.
  Six flags flipping across four modules, with gating read at each call site, is
  the same scattered-policy shape that produced the tier-2 filter-bypass bug. A
  single resolved profile (Stage 1) is the smaller long-run surface.
- **Env-wins precedence (`env > config`)** — considered for benchmark ergonomics
  (a sweep value must not be silently overridden), rejected. It disagrees with the
  existing `RoutingConfig` order and lets a stale shell var silently mask an
  intentional config value — the wrong direction for a product's authoritative
  source. `config > env > default` plus file-level `--config-path` gives
  benchmarks reproducible per-candidate config files instead, and keeps one
  precedence rule across the whole binary.
- **Keep the two indexes separate** (ANN for query, matmul graph for build) —
  rejected. Deriving the graph from the ANN index removes the O(N²) build and one
  whole index-maintenance path.
