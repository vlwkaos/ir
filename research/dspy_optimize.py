#!/usr/bin/env python3
# DSPy prompt optimization for Qwen3.5 expansion + reranking.
# Optimizes against NFCorpus/SciFact qrels using MIPROv2 (expansion) and
# BootstrapFewShot (reranking). Extracts prompts for hardcoding in src/llm/qwen.rs.
#
# Prerequisites:
#   pip install dspy ollama
#   ollama pull qwen3.5:0.8b
#   ollama pull qwen3.5:2b
#   python research/export_eval_data.py   # generates research/artifacts/*.jsonl
#
# Usage:
#   python research/dspy_optimize.py [--model 0.8b|2b] [--task expand|rerank|both]
#
# Outputs:
#   research/artifacts/{model}_expander.json
#   research/artifacts/{model}_reranker.json
#   research/artifacts/{model}_prompts.txt   ← paste into qwen.rs

import argparse
import json
from pathlib import Path

import dspy

ARTIFACTS = Path(__file__).parent / "artifacts"

# ── DSPy signatures ─────────────────────────────────────────────────────────

class ExpandQuery(dspy.Signature):
    """Generate search sub-queries for document retrieval.
    Output exactly three lines starting with 'lex:', 'vec:', 'hyde:'.
    lex: 2-5 keywords or phrases for BM25 keyword search.
    vec: natural language reformulation for semantic/vector search.
    hyde: 1-2 sentence hypothetical answer passage."""
    query: str = dspy.InputField()
    output: str = dspy.OutputField(
        desc="three lines: 'lex: ...', 'vec: ...', 'hyde: ...'"
    )


class JudgeRelevance(dspy.Signature):
    """Judge if the document is relevant to the query.
    Answer only 'Yes' or 'No'."""
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
    """Parse lex/vec/hyde lines from model output."""
    result = {}
    for line in output.splitlines():
        line = line.strip()
        for prefix in ("lex:", "vec:", "hyde:"):
            if line.startswith(prefix):
                result[prefix[:-1]] = line[len(prefix):].strip()
    return result


def expansion_format_metric(example, pred, trace=None):
    """Reward: all three keys present + non-empty."""
    parsed = parse_expansion(pred.output)
    return int(
        "lex" in parsed and "vec" in parsed and "hyde" in parsed
        and all(len(v) > 3 for v in parsed.values())
    )


def rerank_accuracy_metric(example, pred, trace=None):
    """Reward: predicted relevance matches gold."""
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
                query=r["query"],
                document=r["document"],
                relevant=r["relevant"],
            ).with_inputs("query", "document")
            for r in records
        ]

    return load(train_path), load(dev_path)


# ── Optimization ─────────────────────────────────────────────────────────────

def optimize_expander(lm, model_name: str, train_data, dev_data):
    print(f"\n[expand] optimizing {model_name} on {len(train_data)} examples...")
    with dspy.context(lm=lm):
        program = ExpanderProgram()
        optimizer = dspy.MIPROv2(
            metric=expansion_format_metric,
            num_threads=4,
            auto="light",  # light=faster, medium/heavy for production
        )
        optimized = optimizer.compile(
            program,
            trainset=train_data,
            num_trials=20,
            max_bootstrapped_demos=3,
        )

    out_path = ARTIFACTS / f"{model_name}_expander.json"
    optimized.save(str(out_path))
    print(f"  saved → {out_path}")

    # Eval on dev
    with dspy.context(lm=lm):
        scores = [
            expansion_format_metric(ex, optimized(query=ex.query))
            for ex in dev_data[:50]
        ]
    print(f"  dev format-score: {sum(scores)/len(scores):.2%} ({sum(scores)}/{len(scores)})")

    return optimized


def optimize_reranker(lm, model_name: str, train_data, dev_data):
    print(f"\n[rerank] optimizing {model_name} on {len(train_data)} examples...")
    with dspy.context(lm=lm):
        program = RerankerProgram()
        optimizer = dspy.BootstrapFewShot(
            metric=rerank_accuracy_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
        optimized = optimizer.compile(program, trainset=train_data)

    out_path = ARTIFACTS / f"{model_name}_reranker.json"
    optimized.save(str(out_path))
    print(f"  saved → {out_path}")

    with dspy.context(lm=lm):
        scores = [
            rerank_accuracy_metric(ex, optimized(query=ex.query, document=ex.document))
            for ex in dev_data[:50]
        ]
    print(f"  dev accuracy: {sum(scores)/len(scores):.2%} ({sum(scores)}/{len(scores)})")

    return optimized


def extract_prompts(expander_program, reranker_program, model_name: str):
    """Extract optimized system prompts for hardcoding in qwen.rs."""
    out_path = ARTIFACTS / f"{model_name}_prompts.txt"
    with open(out_path, "w") as f:
        f.write(f"# Optimized prompts for {model_name}\n")
        f.write("# Paste into src/llm/qwen.rs constants\n\n")

        # DSPy stores the optimized instructions in the predict modules
        for name, module in [
            ("expander", expander_program),
            ("reranker", reranker_program),
        ]:
            f.write(f"## {name}\n")
            try:
                # Access the underlying predict instructions
                predictor = list(module.predictors())[0]
                if hasattr(predictor, "extended_signature"):
                    f.write(f"signature: {predictor.extended_signature}\n")
                if hasattr(predictor, "demos") and predictor.demos:
                    f.write(f"demos ({len(predictor.demos)}):\n")
                    for i, demo in enumerate(predictor.demos[:3]):
                        f.write(f"  [{i}] {json.dumps(demo, default=str)[:200]}\n")
            except Exception as e:
                f.write(f"(extraction failed: {e})\n")
            f.write("\n")

    print(f"\nPrompts extracted → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.8b", "2b", "both"], default="both")
    parser.add_argument("--task", choices=["expand", "rerank", "both"], default="both")
    parser.add_argument("--dataset", default="nfcorpus")
    args = parser.parse_args()

    models = {
        "0.8b": ("ollama_chat/qwen3.5:0.8b", "qwen35_0.8b"),
        "2b": ("ollama_chat/qwen3.5:2b", "qwen35_2b"),
    }

    target_models = list(models.keys()) if args.model == "both" else [args.model]

    expand_train, expand_dev = load_expand_data(args.dataset)
    rerank_train, rerank_dev = load_rerank_data(args.dataset)

    for model_key in target_models:
        model_id, model_name = models[model_key]
        print(f"\n{'='*60}")
        print(f"Optimizing: {model_id}")
        print(f"{'='*60}")

        lm = dspy.LM(model_id, api_base="http://localhost:11434", temperature=0.7)

        expander_program = reranker_program = None

        if args.task in ("expand", "both"):
            expander_program = optimize_expander(lm, model_name, expand_train, expand_dev)

        if args.task in ("rerank", "both"):
            reranker_program = optimize_reranker(lm, model_name, rerank_train, rerank_dev)

        if expander_program and reranker_program:
            extract_prompts(expander_program, reranker_program, model_name)

    print("\n\nDone. Compare model scores, then hardcode winning prompts in src/llm/qwen.rs.")
    print("Mark them with: // ! DSPy-optimized prompt — do not edit manually")


if __name__ == "__main__":
    main()
