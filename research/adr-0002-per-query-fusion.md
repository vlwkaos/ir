# ADR-0002: Per-query hybrid fusion

Status: accepted. Supersedes the fixed tier-1 fusion of
[ADR-0001](adr-0001-default-retrieval-pipeline.md). Full experiment log, arms and
scripts: branch `research/multiscript-first-stage`, `research/mlir-hypotheses.md`.

## Problem

Tier 1 scored every passage `0.8 · vector + 0.2 · BM25`. BM25 can only score
passages that share the query's words, so in a collection with several languages it
lifts same-language near misses above an answer written in another language. The
vector score gap between them is usually far smaller than the 0.2 BM25 bonus.

## Decision

`score_fusion_two_list` (`src/search/hybrid.rs`) picks one weighting per query:

| Plan | When | Score |
|---|---|---|
| `Keyword` | one of the BM25 top 3 holds every query unit (Latin words, CJK bigrams) | vector, plus 0.2 · BM25 only for passages holding every unit |
| `CrossScript` | else a top-10 vector hit has no main script (≥ 20% of letters) in common with the query | vector only |
| `Weighted` | otherwise | 0.8 · vector + 0.2 · BM25 (unchanged) |

BM25 and the query embedding run in parallel threads.

## Evidence (nDCG@10, paired bootstrap 95% CI)

Test set: MIRACL dev ko/ja/zh/en mixed into one corpus (19,358 passages; ids prefixed
by language), 800 questions (half translated across languages), 270 synthetic
keyword queries; tuning on a dev half, one confirmation run on the test half.

| Set | Before → after | Difference |
|---|---|---|
| Cross-language questions, tier 1 (200) | 0.21 → 0.78 | |
| All mixed questions, tier 1 (400) | 0.53 → 0.82 | +0.283 [+0.245, +0.320] |
| Cross-language questions, with reranker (200) | 0.71 → 0.80 | +0.086 [+0.058, +0.116] |
| Keyword queries (135) | 0.93 → 0.94 | +0.005 [−0.013, +0.022] |
| MIRACL-ko 50k / NFCorpus / FiQA | | identical / +0.002 / −0.003 |

The gain holds with bge-m3 as embedder (+0.256) and with a dictionary-free CJK bigram
BM25 (+0.303). Tier-1 median time fell from about 67 ms to 50 ms (parallel BM25 and
embedding; load average 4–5, indicative).

## Consequences

- In a small collection that mixes languages, more questions take the vector-only
  plan, because a passage in another script is easily among the top 10.
- The reranker adds little on cross-language questions after this change
  (+0.011 [−0.018, +0.040]) while costing about 4 s; a cheaper reranker is the next lever.
- Rejected: vector-only fusion (hurts keyword lookups by 0.05–0.27), routing by BM25
  coverage alone (−0.008 on MIRACL-ko), coverage-gated BM25 for every query (loses
  partial matches on medium-length same-language questions: FiQA −0.009).
