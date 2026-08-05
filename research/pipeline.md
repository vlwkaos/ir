
# ir — Search Pipeline

## Pipeline at a glance (0.18 defaults)

Three-tier escalation. Each tier runs only if the previous tier's signal isn't
strong enough, so cheap queries stay cheap. 0.18 defaults: ANN on, tier-0 graph
expansion on, rerank window 100 + keep-window on, LLM expander off (delegated to
the caller).

```mermaid
flowchart TD
    Q(["ir search &quot;query&quot;"]) --> PRE["Preprocess · ko/ja/zh<br/>lindera · optional"]
    PRE --> T0

    subgraph S0["TIER 0 — no model · in-process · instant"]
        direction TB
        T0["BM25 · FTS5"] --> T0G["+ tier-0 graph expand<br/>doc-graph neighbours · NEW in 0.18"]
    end

    subgraph S1["TIER 1 — embedder · ~50–280ms"]
        direction TB
        T1["Vector kNN · HNSW ANN<br/>exact fallback · NEW in 0.18"] --> FUSE["Hybrid fusion<br/>0.80·vec + 0.20·bm25"]
    end

    subgraph S2["TIER 2 — reranker 0.6B · ~2s"]
        direction TB
        RR["Rerank · window 100 + keep-window<br/>NEW in 0.18"] --> BLEND["blend<br/>0.40·fused + 0.60·P(rel)"]
    end

    T0G -->|"BM25 strong<br/>score≥0.75 ∧ gap≥0.10"| DONE([" ◎ results "])
    T0G -->|escalate| T1
    FUSE -->|"fused strong<br/>top·gap≥0.06 ∧ top≥0.40"| DONE
    FUSE -->|escalate| RR
    BLEND --> DONE

    EXP["LLM expander 1.7B<br/>OFF by default → caller expands"] -. opt-in .-> RR

    classDef t0 fill:#e8f6ee,stroke:#3f9d68,stroke-width:1px,color:#173a26;
    classDef t1 fill:#e6f0fb,stroke:#3f7fc0,stroke-width:1px,color:#14304f;
    classDef t2 fill:#e9e7fb,stroke:#6a5fc0,stroke-width:1px,color:#241f4f;
    classDef done fill:#12897a,stroke:#0c5f54,stroke-width:1px,color:#ffffff;
    classDef off fill:#f6ece0,stroke:#c96a1e,stroke-width:1px,color:#5a3410,stroke-dasharray:4 3;
    class T0,T0G t0;
    class T1,FUSE t1;
    class RR,BLEND t2;
    class DONE done;
    class EXP off;
```

## Staged Async Daemon Design

BM25 runs in-process immediately. Daemon starts in background and signals readiness in two tiers. Client escalates only as far as the query needs.

```
ir search "query"
        │
        ▼
┌─────────────────────┐
│  BM25  (FTS5)       │  no model · in-process · instant
└──────────┬──────────┘  daemon starts in background
           │
     strong signal?         score ≥ 0.75  ∧  gap ≥ 0.10
     ───────yes─────────────────────────────────────────► return
                                               daemon warms in background
           │ no
           ▼
  ┌─────────────────────────────┐
  │  wait: Tier 1               │
  │  Embedder (EmbeddingGemma)  │  ~1s on M-series
  └──────────┬──────────────────┘
           │
┌──────────┴──────────┐
│  Hybrid Score-Fusion │  0.80·vec + 0.20·bm25  →  fused
└──────────┬──────────┘
           │
     strong enough?  ──yes──────────────────────────────► return
     no expander?    ──yes──────────────────────────────► return
                                               daemon finishes loading
           │ no
           ▼
  ┌────────────────────────────────────────────┐
  │  wait: Tier 2                              │
  │  Expander (qmd-1.7B) + Reranker (0.6B)    │  ~3–5s on M-series
  └──────────┬─────────────────────────────────┘
           │
┌──────────┴──────────┐
│  Query Expansion    │  original query → lex / vec / hyde sub-queries
└──────────┬──────────┘
           │
           ├─── lex sub-queries ──► BM25 (FTS5) ─────────────────┐
           │                                                       │
           ├─── vec sub-queries ──► kNN (batch embed) ────────────┤
           │                                                       │
           ├─── hyde sub-queries ─► kNN (batch embed) ────────────┤
           │                                                       │
           └─── fused (from score-fusion above) ──────────────────┤
                                                                   │
                                                                   ▼
                                                        ┌─────────────────┐
                                                        │   RRF merge     │
                                                        └────────┬────────┘
                                                                 │
                                                        ┌────────┴────────┐
                                                        │   Reranking     │  top-20
                                                        │                 │  0.40·fused + 0.60·P(relevant)
                                                        └────────┬────────┘
                                                                 │
                                                                 ▼
                                                              results
```

### Tier Model Requirements

| Tier | Models | Enables |
|------|--------|---------|
| 0 (instant) | none | BM25 only |
| 1 | Embedder | Vector, hybrid score-fusion |
| 2 | Expander + Scorer | Query expansion + reranking |

Note: expander without scorer is a no-op (expansion skipped if no reranker — `hybrid.rs:112`).

### Strong-Signal Shortcut

Raw BM25 score ≥ 0.75 AND gap ≥ 0.10 → skip all LLM work. Implemented in `src/search/hybrid.rs:is_bm25_strong_signal`. Fires rarely on non-English corpora (BM25 near-zero for Korean etc.) — those always escalate to Tier 1 minimum.

---

## Implementation

Staged async model load, two readiness signals:

```
embedder load → bind socket → write tier1 (PID) → [background] expander+reranker load → write tier2
```

Client waits up to 3s for socket (tier 1), then up to 7s for tier2 signal if hybrid mode needs it.

---

## Schema

Each collection DB (`~/.config/ir/collections/<name>.sqlite`):

```
content          — hash → full text (content-addressed, SHA-256)
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
