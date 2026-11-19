# Tier-2 reranker evaluation

Measures a reranker apart from retrieval: each query's tier-2 window is frozen once from a real
hybrid run, and every reranker config ranks the same windows. Runs go on GitHub x64 runners
(`eval-matrix.yml`, research/eval-compute.md).

## Method

```mermaid
flowchart LR
  D["mode=dump: hybrid run<br/>IR_RERANK_DUMP"] --> F["dump.jsonl<br/>window, texts, fused,<br/>pointwise score, ms, tokens"]
  F --> P["pw100 / pw50 / pw20<br/>--from-dump, no model"]
  F --> R["mode=rerank configs=...<br/>ir rerank-eval + model"]
  P --> T["one row per config<br/>.blend and .rerank"]
  R --> T
  F --> O["fused, oracle&lt;w&gt;"]
```

1. `mode=dump` runs `bench.sh --mode hybrid` with `IR_RERANK_DUMP`. Each tier-2 query appends one
   line from `hybrid::rerank`: the window (path, text, fused score, pointwise score, ms, tokens) and the
   tail. The llm_cache is not read, so every cost is measured. A strong-signal shortcut appends a gate
   line (`bm25-strong`, `fused-strong`).
2. `beir-eval.py rerank` refuses the run if a query has no line: a fallback answered it (cold daemon,
   tier-2 timeout, failed query). It also refuses the run if the dump's own scores do not give the
   run's ranking.
3. Per config: `<name>.blend` is ir's final order (0.4·fused + 0.6·score, keep-window).
   `<name>.rerank` is the rerank-score order. `fused` is the window without reranking.
   `oracle<w>` is the best order of the first w docs, the ceiling of that window.
4. Gated queries keep the dump run's ranking in every row. `ndcg_10_t2` and `recall_10_t2` cover
   only the tier-2 queries, and `t2_mean_ms` / `t2_mean_tokens` give the cost per reranked query.

The configs are in `scripts/rerank-configs.json`. A listwise config sets `IR_LISTWISE_MODEL`
(a file in `models`, pinned by HF revision and sha256) and the `IR_LISTWISE_*` knobs. The seed
job caches only the models of the run's `configs` (all of them pass the 10 GB actions cache
limit), so rank one or two listwise models per run. `research_only` models (d1-3B, LFM Open
License v1.0) give benchmark rows only: ir never defaults to or downloads them.

## How to run

```bash
gh workflow run eval-matrix.yml --ref <branch> -f dataset=scifact -f mode=dump -f shards=20
gh workflow run eval-matrix.yml --ref <branch> -f dataset=scifact -f mode=rerank -f shards=20 \
  -f configs=lw-qwen3-0.6b-n20-t160
```

`mode=rerank` restores the dump of the same `DUMP_KEY`: the index, the reranker and the
retrieval code. After a change to any of them, run `mode=dump` again.

Local replay of one config over a dump:

```bash
IR_LISTWISE_MODEL=<gguf> ir rerank-eval --input dump.jsonl --output out.jsonl --window 20
```

## Listwise scorer (`src/llm/listwise.rs`)

N passages go in one prompt, and one forward pass scores them all (FIRST, arXiv 2406.15657). A
window longer than N slides from the tail (RankGPT). The score is 1 - rank/M. Pointwise stays the
default. The GGUF's `<arch>.decision.type` picks the model family (research/decision-models.md).
A family is three parts: the prompt layout (`Layout`), how a score is read (`Readout`), and the
limits it declares (`max_n`, and the default passage cut `Family::trunc`). The window, shuffle
and context code reads only these, so a new family is one `Layout` variant and one `detect` arm,
plus a `Readout` variant only when it reads scores a new way.

| Type | Models | Prompt | Readout | Max N |
|---|---|---|---|---|
| none | instruct (Qwen3) | the GGUF's chat template, `[A] passage` | logit of `A` at the last token | 26 |
| `vllm-d3` | d3-lite, d3-nano | d3_format.py: system prompt, `State:` query, `A: passage` | logit of `A` | 26 |
| `lfm2-d1` | d1-3B | prompt.py: no system, `QUESTION:`, `A passage` | max of `A`, ` A` | 26 |
| `laya` | Laya (ModernBERT encoder) | `[CLS]choice question: Q[SEP]`, `[MASK] passage` per option, `[SEP]Search query: q[SEP]` | decision head column 0 at each `[MASK]`, softmax with the GGUF's temperature for that many options | 20 |

The Laya prompt is a port of llama.cpp d812350: the GGUF template
`tokenizer.chat_template.systemone` (written by `conversion/bert.py`) and
`tools/server/server-decision.cpp` `fill_task_laya`. The question and all options share the head's
`decision.max_head_tokens` (192 for English Laya, 256 for laya-multilingual). Each option keeps at
most 48 tokens after its marker; when the options do not fit, each keeps
max(4, (head - 16) / N) tokens with its marker. So at N = 20 an English option keeps 7 passage
tokens and a multilingual one 11. Without `IR_LISTWISE_TRUNC`, Laya cuts passages to 48 tokens
first, and the head budget then cuts them for the number of options in each prompt.
The passage's own text replaces the option label: Laya needs no label, and the label would cost
head tokens.

llama-cpp-2 is 0.1.160, the first release that bundles llama.cpp d812350, which has the decision
head (PR 29818). libllama applies the GGUF `attention.causal = false` key itself (d3's
full-attention layers, ModernBERT). So every prompt decodes in one batch (`n_ubatch` = context):
with `n_ubatch` 64, d3-lite hits llama.cpp's assert "non-causal attention requires n_ubatch >=
n_tokens", and Qwen3 does not.

Observed in development (Qwen3-0.6B Q8, Metal): the model favors label A strongly. With two
passages it picks A in both orders (logit 34.5 vs 32.8), and `IR_LISTWISE_SHUFFLES=2` fixes
that case. One 5-passage prompt takes about 80-200 ms.

### llama-cpp-2 0.1.145 to 0.1.160: speed and scores

20 frozen scifact tier-2 windows, N = 20, `IR_LISTWISE_TRUNC=160`, window 20, Mac Metal, 3
interleaved runs per row. `before` = main's listwise.rs on 0.1.145; `refactor` = the family
split on 0.1.145; `split + 0.1.160` = the family split on 0.1.160. Max |Δscore| and same order
compare with the first `before` run. This table mixes the two changes; the next one separates them.

| Model | Binary | Max \|Δscore\| | Same order | Median ms / window (min-max) |
|---|---|---|---|---|
| Qwen3-0.6B | before | 0 | 20/20 | 2069 (2001-2076) |
| Qwen3-0.6B | refactor | 0 | 20/20 | 1872 (1843-2002) vs 1933 (1867-1968) |
| Qwen3-0.6B | split + 0.1.160 | 0.05 | 17/20 | 2025 (1981-2154) |
| d3-lite | before | 0 | 20/20 | 3622 (3018-3682) |
| d3-lite | refactor | 0 | 20/20 | 2772 (2733-2807) vs 2748 (2651-2931) |
| d3-lite | split + 0.1.160 | 0.05 | 18/20 | 2950 (2633-3510) |
| d1-3B | before | 0 | 20/20 | 34780 (25229-36213) |
| d1-3B | split + 0.1.160 | 0.05 | 19/20 | 30610 (28314-37551) |

- 0.05 = 1/20 is one adjacent rank swap. It comes from the new llama.cpp kernels, not ir code.

Family split alone on 0.1.160: 759b681 (the bump, main's listwise.rs) against bebe29a (the
split), the same 20 windows and settings, 3 runs each, the two binaries interleaved and their
order swapped per run. Max |Δscore| and same order compare with the first 759b681 run. d1-3B
not run (about 9 min per run).

| Model | Binary | Max \|Δscore\| | Same order | Median ms / window (min-max) | Tokens |
|---|---|---|---|---|---|
| Qwen3-0.6B | 759b681 | 0 | 20/20 ×3 | 1901 (1874-1943) | 65536 |
| Qwen3-0.6B | bebe29a | 0 | 20/20 ×3 | 1891 (1866-1925) | 65536 |
| d3-lite | 759b681 | 0 | 20/20 ×3 | 2591 (2548-2610) | 65965 |
| d3-lite | bebe29a | 0 | 20/20 ×3 | 2614 (2563-2623) | 65965 |

- The family split changes no score and no prompt token. Its speed is within the run spread:
  -0.5% for Qwen3 and +0.9% for d3-lite, and the ranges overlap. The `refactor` rows show the
  same on 0.1.145.
- Laya (English, 449 MB) ranks one 20-passage scifact window in 327 ms with 183 tokens.

### laya-multilingual conversion

`convert_hf_to_gguf.py` at d812350 (its pinned transformers 4.57.6) fails on
convaiinnovations/laya-multilingual rev 92d22143, after the tensors export:

```
conversion/base.py:1588 AutoTokenizer.from_pretrained(self.dir_model)
transformers/tokenization_utils_base.py:1210
AttributeError: 'list' object has no attribute 'keys'
```

First cause: `tokenizer/tokenizer_config.json:6` writes `extra_special_tokens` as a list, the
transformers 5.0.0 format (`encoder/config.json:77`). Transformers 4.57.6 expects a dict. No
GGUF, so no miracl-ko row for it.

## Results

Proof gates for a dump run: its hybrid row equals the trusted hybrid run ±0.001. `pw100.blend`
equals the hybrid row. `oracle100` is at least every other row. Stop a listwise model at the first
dataset where it loses to `pw20` on both nDCG@10 and ms/query.

### scifact

Dump run 43999825138 (f38596b); listwise runs 44016959846 (d3, 62afb6f) and 44024239967 (d1-3B, 62afb6f), 44036692153 (Qwen3 control, 4c342f8). 300 queries, 183 reach
tier 2, no failed query. ms and tokens are per tier-2 query on one GitHub x64 runner.

| Row | nDCG@10 | R@10 | nDCG@10 t2 | ms / t2 query | tokens / t2 query |
|---|---|---|---|---|---|
| fused | 0.7827 | 0.8949 | 0.6664 | 0 | 0 |
| hybrid (dump run) | 0.7806 | 0.8977 | | | |
| pw100.blend | 0.7806 | 0.8977 | 0.6629 | 242540 | 36385 |
| pw50.blend | 0.7761 | 0.8943 | 0.6555 | 122552 | 18283 |
| pw20.blend | 0.7773 | 0.8932 | 0.6576 | 49996 | 7329 |
| lw-qwen3-0.6b-n20-t160.blend | 0.7645 | 0.8645 | 0.6365 | 34016 | 3294 |
| lw-d3-lite-n20-t160.blend | 0.7370 | 0.8482 | 0.5915 | 32321 | 3316 |
| lw-d3-nano-n20-t160.blend | 0.7398 | 0.8432 | 0.5960 | 67389 | 3316 |
| lw-d1-3b-n20-t160.blend | 0.7653 | 0.8806 | 0.6379 | 104248 | 3247 |
| oracle20 / 50 / 100 | 0.8962 / 0.9249 / 0.9649 | 0.9056 / 0.9350 / 0.9750 | 0.8524 / 0.8995 / 0.9651 | | |

- No reranker beats `fused`: tier 2 lowers scifact nDCG@10 in every config.
- The `.rerank` order (score only) is below `.blend` in every config: pw20 0.7723, d3-lite 0.7261,
  d3-nano 0.7308, d1-3B 0.7555, Qwen3 0.7624.
- The Qwen3-0.6B control (generic instruct, same listwise prompt path) ties d1-3B on nDCG@10
  (0.7645 vs 0.7653) in a third of the time (34 s vs 104 s) and beats both d3 models. The
  listwise method works; the d3 loss is the d3 judge.
- d1-3B is the best decision model (0.7653, 0.012 below pw20), but it takes 104 s per tier-2 query.
  d3-lite is the fastest (32 s, 35% below pw20) and loses 0.040.
- The stop rule would stop d3-nano and d1-3B here. Scifact alone does not decide, because pointwise
  also loses to fused on it. miracl-ko runs the pw rows and d1-3B, the best decision model (its card
  names Korean and lists reranking as an intended use).

The d3 readout was checked before the d3 rows were accepted:

1. Position bias: none. In the 183 tier-2 windows, d3-lite picks label A for 38% of queries,
   B 11%, C 8%, and no later label above 5%; T gets 1%. The relevant docs sit at A for 50% of
   queries (92 of 190 relevant docs in the windows are at A). So d3 picks A less often than the
   fused order would. Its non-A picks are relevant in 21 of 113 queries (d3-nano: 14 of 96).
2. Label readout: correct. A replay of 10 tier-2 queries (`ir rerank-eval`, Metal) read labels
   A..T as single tokens 32..51, at the last prompt token after `<think>\n\n</think>\n\n`. This
   matches d3_format.py (`d3-code-readout-v1`: `A: text` options, bare code tokens 32..57, readout
   at the last position). The argmax over the whole vocabulary was the chosen label in 10 of 10,
   and the top pick equals the CI x64 run's in 10 of 10.
3. `IR_LISTWISE_SHUFFLES=2` was not run: 1 and 2 show no bias. The d3 cards present d3 for routing
   and classification decisions, not for ranking. Its rerank order is below fused because the judge
   is weak on scifact, not because of the readout.

### miracl-ko (10000-doc pool)

Dump run 44060568465 and listwise run 44070074345 (both 3b1d6c5). 213 queries, 136 reach tier 2,
no failed query.

| Row | nDCG@10 | R@10 | nDCG@10 t2 | ms / t2 query | tokens / t2 query |
|---|---|---|---|---|---|
| fused | 0.9539 | 0.9805 | 0.9432 | 0 | 0 |
| hybrid (dump run) | 0.9737 | 0.9890 | | | |
| pw100.blend | 0.9737 | 0.9890 | 0.9742 | 117279 | 17542 |
| pw50.blend | 0.9731 | 0.9881 | 0.9732 | 61089 | 8975 |
| pw20.blend | 0.9680 | 0.9826 | 0.9652 | 27163 | 3797 |
| lw-qwen3-0.6b-n20-t160.blend | 0.9138 | 0.9274 | 0.8804 | 25584 | 2641 |
| lw-d3-lite-n20-t160.blend | 0.9315 | 0.9598 | 0.9081 | 23104 | 2393 |
| lw-d1-3b-n20-t160.blend | 0.9487 | 0.9795 | 0.9350 | 80225 | 2406 |
| oracle20 / 50 / 100 | 0.9838 / 0.9896 / 0.9902 | 0.9844 / 0.9908 / 0.9917 | 0.9899 / 0.9990 / 1.0 | | |

- Pointwise beats fused here (+0.020 at window 100, +0.014 at window 20). Every listwise row loses
  to fused. `.rerank` rows: d1-3B 0.9445, d3-lite 0.9157, Qwen3 0.8956.
- d1-3B beats the Qwen3 control by 0.035 (scifact: tie). It loses to pw20 on nDCG@10 and ms, so the
  stop rule stops it. d3-lite and Qwen3 are only 5-15% faster than pw20: the ko passages are short.

### Passage cap (`IR_LISTWISE_TRUNC=160`)

Pointwise reads a passage up to 6000 chars (`src/llm/scoring.rs:16`); listwise keeps its first 160
tokens (`src/llm/listwise.rs:104`). Top-20 window docs in the dumps, Qwen3 tokenizer (per-doc
pointwise tokens minus the 9-token suffix):

| Dataset | tokens / doc (mean / median) | docs over 160 | doc tokens kept |
|---|---|---|---|
| scifact | 354 / 324 | 96% | 45% |
| miracl-ko | 176 / 147 | 44% | 72% |
| fiqa | 257 / 196 | 59% | 53% |

The d1-3B lead over the Qwen3 control grows with the share of tokens kept: scifact 45% (tie),
fiqa 53% (+0.025), miracl-ko 72% (+0.035). fiqa is English, so Korean alone does not explain the
lead. This fits the cap as a ceiling for d1-3B, but does not prove it. A t320 run on scifact
separates the causes.

### fiqa (10000-doc pool)

Dump run 44090564157 (63c3be0) and listwise run 44142388088 (77d031d). 648 queries, 541 reach
tier 2, no failed query.

| Row | nDCG@10 | R@10 | nDCG@10 t2 | ms / t2 query | tokens / t2 query |
|---|---|---|---|---|---|
| fused | 0.5934 | 0.6579 | 0.5513 | 0 | 0 |
| hybrid (dump run) | 0.6275 | 0.6962 | | | |
| pw100.blend | 0.6275 | 0.6962 | 0.5921 | 182296 | 26861 |
| pw50.blend | 0.6208 | 0.6864 | 0.5840 | 93094 | 13694 |
| pw20.blend | 0.6165 | 0.6760 | 0.5789 | 36762 | 5380 |
| lw-qwen3-0.6b-n20-t160.blend | 0.5652 | 0.6188 | 0.5175 | 27279 | 2804 |
| lw-d3-lite-n20-t160.blend | 0.5007 | 0.5926 | 0.4402 | 27690 | 2844 |
| lw-d1-3b-n20-t160.blend | 0.5899 | 0.6646 | 0.5471 | 90773 | 2768 |
| oracle20 / 50 / 100 | 0.7399 / 0.7845 / 0.8489 | 0.7074 / 0.7555 / 0.8303 | 0.7267 / 0.7802 / 0.8572 | | |

- Pointwise beats fused (+0.034 at window 100, +0.023 at window 20). The gain comes from the blend:
  `.rerank` pw100 0.5982, pw50 0.5912, pw20 0.5925.
- Every listwise row loses to fused. `.rerank` rows: d1-3B 0.5818, Qwen3 0.5620, d3-lite 0.4829.
- d1-3B beats the Qwen3 control by 0.025. It loses to pw20 on nDCG@10 (0.027) and ms (2.5x), so the
  stop rule stops it. Qwen3 and d3-lite are 25% faster than pw20 and lose 0.051 and 0.116.

### Verdict

No listwise row reaches pw20 on any of the three datasets, and the stop rule stops every decision
model. Pointwise stays the default. Listwise reads 45-70% of the tokens of pw20 per query. On a
4-CPU x64 runner Qwen3 and d3-lite are 6-35% faster than pw20, and d1-3B is 2-3x slower.
