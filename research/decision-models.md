# Local decision models as ir's reranker (scout, 2026-11-17)

Answer: benchmark **vllm-sr/d3-lite** (0.85B) first, then **LiquidAI/d1-3B** (3.1B, Korean named, reranking an intended use),
then **vllm-sr/d3-nano** (2.2B). Their GGUF readout is plain LM-head label-token logits, which matches task #22's design
(N labeled passages, label logits from one forward pass through llama-cpp-2). Most other decision models put a custom head
in the graph or outside it, which llama-cpp-2 cannot reach today.

Scope: read-only. Nothing was downloaded or run. All scores below are **self-reported** by the model authors. Downloads
are the Hub's last-30-day count, queried 2026-11-17. This updates the 2026-11-10 scan in wsx notes
`sag-wiki-review.md`, section "Local decision models", and does not repeat it.

## The constraint that decides fit: llama-cpp-2 bundles an old llama.cpp

| Crate | Bundled llama.cpp commit | Commit date |
|---|---|---|
| llama-cpp-2 0.1.145 (#22's target) | `e21cdc1` | 2026-04-13 |
| llama-cpp-2 0.1.161 (latest, 2026-11-03) | `d812350` | 2026-10-05 |

0.1.145 predates every decision-model PR. 0.1.161 contains only the first two: PR 29818 (decision-head graph, Laya) and
PR 29831 (Clef), checked with `gh api .../compare`. Every later PR (pplx-decider, d1, d3, h2o, decision2) came after it.
`/v1/systemone` and each model's prompt protocol live in `tools/server`, not in libllama.
So ir can use a decision model only if both of these are true:
1. Its readout is the LM-head logits of label tokens.
2. Its architecture already exists at `e21cdc1`. That commit has `qwen35`, `lfm2` and `modern-bert`, but not `lfm3`.

llama.cpp's own list of decision types (`common/common.h`, `enum common_decision_type`) sorts the models:

| Readout in stock llama.cpp | Decision types | Reachable from llama-cpp-2 0.1.145 |
|---|---|---|
| Label-token logits, causal | openjev, lev, nimble, pplx-decider (v1), lfm2-d1, h2o-lightning (one-way) | yes, with `get_logits_ith` |
| Label-token logits, non-causal | vllm-d3, lfm3-d1 | vllm-d3: yes, with a `llama_set_causal_attn(false)` sys call (unchecked on qwen35 at `e21cdc1`). lfm3-d1: no, the `lfm3` arch is missing |
| Score read from the embeddings or hidden states | laya, clef, vllm-decision2, lfm2-d1-omni, kev, minicpm-ocr-jev | no: needs the decision-head graph (PR 29818; in 0.1.161, not 0.1.145) or a later PR |

## Shortlist (2B or less, plus a 4B stretch; GGUF or a small encoder)

All rows pick among N options (up to 255, except Laya at about 20). "Lang" means the languages the card states.

| Hub id (original) | Created | DL | Params | Base | License | Lang | Stock llama.cpp | One-pass label logits via crate |
|---|---|---|---|---|---|---|---|---|
| [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) | 2026-10-10 | 5,499 | 0.59B | Qwen3.5-0.8B | Apache-2.0 | not stated | yes: vllm-d3, PR 30383, `ggml-org/vllm-d3-edge-GGUF` | yes, non-causal |
| [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite) | 2026-10-10 | 6,821 | 0.85B | Qwen3.5-0.8B | Apache-2.0 | not stated | yes: `ggml-org/vllm-d3-lite-GGUF` | yes, non-causal |
| [vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano) | 2026-10-10 | 7,411 | 2.21B (0.33B vision) | Qwen3.5-2B | Apache-2.0 | not stated | yes: `ggml-org/vllm-d3-nano-GGUF` | yes, non-causal |
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (stretch) | 2026-10-05 | 22,200 | 3.12B | LFM2.5-VL-3B | LFM Open v1.0 | 16, **ko** | yes: lfm2-d1, PR 30110 (no libllama change), `LiquidAI/d1-3B-GGUF` | yes, causal |
| [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b) (stretch) | 2026-10-02 | 9,852 | about 4.2B text | Qwen3.5-4B | Apache-2.0 | mostly en | yes: h2o-lightning, PR 30361 (adds only `decision.labels` metadata), `h2oai/h2o-lightning-4b-gguf` | yes, one-way (causal) mode; two-way needs h2o's vLLM fork |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 2026-09-19 | 8,778 | 322M | mmBERT-base | Apache-2.0 | 100+, **ko** | English Laya only (`ggml-org/Laya-GGUF`, PR 29818); no multilingual GGUF found | no: custom head |
| [vllm-sr/Decision-2.0-Kai-0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B) | 2026-09-28 | 1,032 | 0.6B | Qwen3-0.6B-Base | Apache-2.0 | not stated | yes: vllm-decision2, PR 30378 | no: head in graph; superseded by d3 |
| [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) / [-2b](https://huggingface.co/Mapika/decider-2b) | 2026-09-19 / 09-16 | 4,468 / 7,674 | 0.75B / 1.88B | Qwen3.5 Base | Apache-2.0 | en only | no decision type; community GGUFs only | likely (the card says "letter logits at an answer slot"); attention mode not stated |
| [tasksource/tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano) | 2026-09-28 | 858 | 150M | ettin-reranker-150m | Apache-2.0 | not stated | no GGUF; custom tree-attention mask | no: linear head |

Self-reported scores. "DI" is the Jev Decision Index, which has a Retrieval area.

| Model | DI (version) | Retrieval or rerank | Latency |
|---|---|---|---|
| d3-edge / d3-lite / d3-nano | 20.2 / 32.3 / 38.9 (0.3.2 official text board) | none reported | nano: 13.4 ms per text request |
| d1-3B | 48.57 (0.2.1) | DI Retrieval area 52.8; "reranking" is a named use | 8 ms on an RTX 4090 |
| h2o-lightning-4b | 63.21 (0.3 public, two-way); 60.86 one-way | TREC-COVID −0.4, not significant | median 25 ms (vLLM) |
| laya-multilingual | n/a | none; MASSIVE Korean intent accuracy 0.490 | 32.8 ms per question |

## Seen, unvetted (younger than 30 days, or under 100 downloads; not recommended)

- **LiquidAI/d1-2B** (2026-10-24, 14,128 DL, 1.92B, LFM3-VL-2B, ko named, reranking named, DI 51.1). It needs the `lfm3` arch and libllama changes (PR 31275, llama.cpp b11960 or later), which no llama-cpp-2 release bundles yet. It is the best candidate once the crate catches up, and becomes vetted by age on 2026-11-23.
- **perplexity-ai/pplx-decider-flash-v1.3** (2026-11-09, 1,027 DL, 4.54B, Apache-2.0, repo private). Protocol v4 is non-causal, with a 255-row readout head. Stock llama.cpp runs only the v1 type. The GGUF ports `vblanco/` and `fxoz/pplx-decider-flash-v1.3-GGUF` need the vblanco fork branch `pplx-decider-v4-macaq`.
- **StrandsAgents/strands-decider-2B-qwen3.5-v1-2610**: the original of the ArminBurkhardt GGUF (2026-10-09, 0 DL as a LoRA adapter, Apache-2.0, en). It uses a pointer head in `head.safetensors`. The GGUF holds only the torso; a numpy client runs N+1 prefills. The source repo `strands-labs/strands-decider` has 1,207 stars.
- **social-jev/social-jev-0.6B, 1.7B and 4B-preview** (2026-11-11, 70 to 207 DL). They state no license, need `trust_remote_code`, and do not describe the readout.
- **Athekunal/ultrabert-jev-380m-softmax-yes and -sigmoid-norm** (2026-11-11, about 70 DL, en, Apache-2.0). They score listwise from the MLM `yes` logit at one `[MASK]` per option in one pass. They have no GGUF.
- **gparitosh007/Open-Decision-Model-150M** (2026-10-31, 107 DL, ModernBERT-base, en). Linear head; the source code is not public.
- **Valen-Team/Valen-Decision-1.5-0.8B and -2B** (2026-11-04, 212 and 260 DL). They need a Valen-patched llama.cpp. The repo has 919 stars.
- **chaoliangUNSW/Jev-Style-2B-Decision-v4.2-GGUF** (18k DL for v4). It needs a pinned llama.cpp fork and a custom scorer, and it is distilled from the TypeSafe Jev API.
- Out of size (9B to 27B): Cloudflare clef-flash (9.4B; `ggml-org/clef-flash-GGUF`, PR 29831), decisionmodels/reax-1.x (27B), Futurix-AI/FxAI-decision-model (26B), vllm-sr d3-flash (8.4B).

## Changes since the 2026-11-10 scan

- New vendor families: LiquidAI d1, vllm-sr d3, h2o-lightning, and the decision-type list in llama.cpp.
- pplx-decider-flash moved from v1.2 to v1.3. Stock llama.cpp does not run v1.3 correctly.
- Mapika/decider is still English only. The mrm8488 typed-decision model is still not stock.

## Open for #22

1. Each decision model is trained on its own prompt protocol. #22 must port the renderer for d3 or d1 from `tools/server/server-decision.cpp`. In that port, the query becomes the question and the passages become the options.
2. Context length for 100 passages: d1-3B has 32k; the d3 cards do not state one.
3. Non-causal attention for d3 on `qwen35` at `e21cdc1` is not verified.
4. License: LFM Open License v1.0 (d1) allows redistribution (section 4). Commercial use is licensed only for entities under $10M annual revenue (section 5). ir would download the weights, not bundle them.
5. None of these was measured against Qwen3-reranker-0.6B, which takes about 90 s per query on a 4-vCPU x64 CPU.

Sources (2026-11-17): Hub API and model cards for each id above; llama.cpp PRs 29818, 29831, 30044, 30110, 30361, 30378, 30383, 31275 and `common/common.h`; utilityai/llama-cpp-rs tags 0.1.145 and 0.1.161; LFM Open License v1.0 (`LiquidAI/d1-2B/LICENSE`); HN "Clef: Open-weight decision models" (2026-10-01), "Strands Decider 2B" (2026-10-07), "Perplexity Decider v1.3" (2026-11-13); [aiweekly: Strands Decider 2B](https://aiweekly.co/alerts/amazon-ships-strands-decider-2b-an-open-source-jev-rival); [aiweekly: Perplexity 27B Decider](https://aiweekly.co/alerts/perplexity-open-sources-27b-decider-edges-jev-on-11-test-panel).
