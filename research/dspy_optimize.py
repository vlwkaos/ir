#!/usr/bin/env python3
# DSPy prompt optimization for Qwen3.5 expansion + reranking.
# Optimizes against NFCorpus/SciFact qrels using MIPROv2 (expansion) and
# BootstrapFewShot (reranking). Extracts prompts for hardcoding in src/llm/qwen.rs.
#
# Prerequisites:
#   pip install dspy beir
#   brew services start ollama
#   ollama pull qwen3.5:0.8b && ollama pull qwen3.5:2b
#   python research/export_eval_data.py
#
# Usage:
#   python research/dspy_optimize.py [--model 0.8b|2b|both] [--task expand|rerank|both]
#   python research/dspy_optimize.py --resume          # skip tasks with existing checkpoints
#
# Live output:
#   tail -f research/artifacts/run.log
#
# Outputs (in research/artifacts/):
#   {model}_expander.json / {model}_reranker.json   — optimized DSPy programs
#   {model}_prompts.txt                             — paste into src/llm/qwen.rs

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import dspy

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

LOG_FILE = ARTIFACTS / "run.log"


def setup_logging():
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Force stdout to be line-buffered so tail -f works
    sys.stdout.reconfigure(line_buffering=True)


log = logging.getLogger(__name__)


# ── DSPy signatures ─────────────────────────────────────────────────────────

class ExpandQuery(dspy.Signature):
    """Generate search sub-queries for document retrieval.
    Output exactly three lines starting with 'lex:', 'vec:', 'hyde:'.
    lex: 2-5 keywords or phrases for BM25 keyword search.
    vec: natural language reformulation for semantic/vector search.
    hyde: 1-2 sentence hypothetical answer passage."""
    query: str = dspy.InputField()
    output: str = dspy.OutputField(desc="three lines: 'lex: ...', 'vec: ...', 'hyde: ...'")


class JudgeRelevance(dspy.Signature):
    """Judge if the document is relevant to the query. Answer only 'Yes' or 'No'."""
    query: str = dspy.InputField()
    document: str = dspy.InputField()
    relevant: bool = dspy.OutputField()


# ── Programs ─────────────────────────────────────────────────────────────────

class ExpanderProgram(dspy.Module):
    def __init__(self):
        self.expand = dspy.ChainOfThought(ExpandQuery)

    def forward(self, query):
        return self.expand(query=query)


class RerankerProgram(dspy.Module):
    def __init__(self):
        self.judge = dspy.Predict(JudgeRelevance)

    def forward(self, query, document):
        return self.judge(query=query, document=document)


# ── Metrics ──────────────────────────────────────────────────────────────────

def parse_expansion(output: str) -> dict:
    result = {}
    for line in output.splitlines():
        line = line.strip()
        for prefix in ("lex:", "vec:", "hyde:"):
            if line.startswith(prefix):
                result[prefix[:-1]] = line[len(prefix):].strip()
    return result


def expansion_format_metric(example, pred, trace=None):
    parsed = parse_expansion(pred.output)
    return int(
        "lex" in parsed and "vec" in parsed and "hyde" in parsed
        and all(len(v) > 3 for v in parsed.values())
    )


def rerank_accuracy_metric(example, pred, trace=None):
    return int(bool(pred.relevant) == bool(example.relevant))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_expand_data(dataset: str = "nfcorpus"):
    train_path = ARTIFACTS / f"{dataset}_expand_train.jsonl"
    dev_path = ARTIFACTS / f"{dataset}_expand_dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Run export_eval_data.py first: {train_path}")

    def load(path):
        with open(path) as f:
            records = [json.loads(l) for l in f]
        return [dspy.Example(query=r["query"]).with_inputs("query") for r in records]

    return load(train_path), load(dev_path)


def load_rerank_data(dataset: str = "nfcorpus"):
    train_path = ARTIFACTS / f"{dataset}_rerank_train.jsonl"
    dev_path = ARTIFACTS / f"{dataset}_rerank_dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Run export_eval_data.py first: {train_path}")

    def load(path):
        with open(path) as f:
            records = [json.loads(l) for l in f]
        return [
            dspy.Example(
                query=r["query"], document=r["document"], relevant=r["relevant"],
            ).with_inputs("query", "document")
            for r in records
        ]

    return load(train_path), load(dev_path)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def checkpoint_path(model_name: str, task: str) -> Path:
    return ARTIFACTS / f"{model_name}_{task}er.json"


def is_done(model_name: str, task: str) -> bool:
    return checkpoint_path(model_name, task).exists()


# ── Optimization ─────────────────────────────────────────────────────────────

def optimize_expander(lm, model_name: str, train_data, dev_data, resume: bool):
    out_path = checkpoint_path(model_name, "expand")

    if resume and out_path.exists():
        log.info("[expand/%s] checkpoint found — loading %s", model_name, out_path)
        program = ExpanderProgram()
        program.load(str(out_path))
        return program

    log.info("[expand/%s] starting — %d train examples", model_name, len(train_data))
    t0 = time.time()

    with dspy.context(lm=lm):
        program = ExpanderProgram()
        # auto="light" controls num_candidates+num_trials internally (DSPy 3.x);
        # cannot combine with explicit num_trials/max_bootstrapped_demos
        optimizer = dspy.MIPROv2(
            metric=expansion_format_metric,
            num_threads=4,
            auto="light",
        )
        log.info("[expand/%s] MIPROv2 compile — this will take several minutes", model_name)
        optimized = optimizer.compile(program, trainset=train_data)

    optimized.save(str(out_path))
    log.info("[expand/%s] saved checkpoint → %s (%.0fs)", model_name, out_path, time.time() - t0)

    with dspy.context(lm=lm):
        log.info("[expand/%s] evaluating on %d dev examples...", model_name, min(50, len(dev_data)))
        scores = [
            expansion_format_metric(ex, optimized(query=ex.query))
            for ex in dev_data[:50]
        ]
    log.info("[expand/%s] dev format-score: %.1f%% (%d/%d)",
             model_name, 100 * sum(scores) / len(scores), sum(scores), len(scores))

    return optimized


def optimize_reranker(lm, model_name: str, train_data, dev_data, resume: bool):
    out_path = checkpoint_path(model_name, "rerank")

    if resume and out_path.exists():
        log.info("[rerank/%s] checkpoint found — loading %s", model_name, out_path)
        program = RerankerProgram()
        program.load(str(out_path))
        return program

    log.info("[rerank/%s] starting — %d train examples", model_name, len(train_data))
    t0 = time.time()

    with dspy.context(lm=lm):
        program = RerankerProgram()
        optimizer = dspy.BootstrapFewShot(
            metric=rerank_accuracy_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
        log.info("[rerank/%s] BootstrapFewShot compile...", model_name)
        optimized = optimizer.compile(program, trainset=train_data)

    optimized.save(str(out_path))
    log.info("[rerank/%s] saved checkpoint → %s (%.0fs)", model_name, out_path, time.time() - t0)

    with dspy.context(lm=lm):
        log.info("[rerank/%s] evaluating on %d dev examples...", model_name, min(50, len(dev_data)))
        scores = [
            rerank_accuracy_metric(ex, optimized(query=ex.query, document=ex.document))
            for ex in dev_data[:50]
        ]
    log.info("[rerank/%s] dev accuracy: %.1f%% (%d/%d)",
             model_name, 100 * sum(scores) / len(scores), sum(scores), len(scores))

    return optimized


def extract_prompts(expander_program, reranker_program, model_name: str):
    out_path = ARTIFACTS / f"{model_name}_prompts.txt"
    with open(out_path, "w") as f:
        f.write(f"# Optimized prompts for {model_name} — {datetime.now().isoformat()}\n")
        f.write("# Paste into src/llm/qwen.rs constants\n\n")

        for task_name, module in [("expander", expander_program), ("reranker", reranker_program)]:
            f.write(f"## {task_name}\n")
            try:
                predictor = list(module.predictors())[0]
                if hasattr(predictor, "extended_signature"):
                    f.write(f"signature: {predictor.extended_signature}\n")
                if hasattr(predictor, "demos") and predictor.demos:
                    f.write(f"demos ({len(predictor.demos)}):\n")
                    for i, demo in enumerate(predictor.demos[:3]):
                        f.write(f"  [{i}] {json.dumps(demo, default=str)[:300]}\n")
            except Exception as e:
                f.write(f"(extraction failed: {e})\n")
            f.write("\n")

    log.info("prompts extracted → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.8b", "2b", "both"], default="both")
    parser.add_argument("--task", choices=["expand", "rerank", "both"], default="both")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--resume", action="store_true",
                        help="skip tasks whose checkpoint already exists")
    args = parser.parse_args()

    log.info("=== dspy_optimize start  model=%s  task=%s  dataset=%s  resume=%s ===",
             args.model, args.task, args.dataset, args.resume)
    log.info("live log: tail -f %s", LOG_FILE)

    models = {
        "0.8b": ("ollama_chat/qwen3.5:0.8b", "qwen35_0.8b"),
        "2b":   ("ollama_chat/qwen3.5:2b",   "qwen35_2b"),
    }
    target_models = list(models.keys()) if args.model == "both" else [args.model]

    expand_train, expand_dev = load_expand_data(args.dataset)
    rerank_train, rerank_dev = load_rerank_data(args.dataset)
    log.info("data loaded: %d expand train, %d rerank train", len(expand_train), len(rerank_train))

    for model_key in target_models:
        model_id, model_name = models[model_key]
        log.info("─── model: %s ───", model_id)

        lm = dspy.LM(model_id, api_base="http://localhost:11434", temperature=0.7)

        # Smoke-test: verify model is listed in ollama before a long run
        import urllib.request
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
                tags = json.load(r)
            names = [m["name"] for m in tags.get("models", [])]
            tag = model_id.split("/")[-1]  # e.g. "qwen3.5:0.8b"
            if not any(tag in n for n in names):
                log.error("model %s not found in ollama (%s) — skipping", tag, names)
                continue
            log.info("smoke ok: %s present in ollama", tag)
        except Exception as e:
            log.error("ollama unreachable: %s — skipping", e)
            continue

        expander_program = reranker_program = None

        if args.task in ("expand", "both"):
            expander_program = optimize_expander(lm, model_name, expand_train, expand_dev, args.resume)

        if args.task in ("rerank", "both"):
            reranker_program = optimize_reranker(lm, model_name, rerank_train, rerank_dev, args.resume)

        if expander_program and reranker_program:
            extract_prompts(expander_program, reranker_program, model_name)

    log.info("=== done — hardcode winning prompts in src/llm/qwen.rs ===")
    log.info("mark with: // ! DSPy-optimized prompt — do not edit manually")


if __name__ == "__main__":
    main()
