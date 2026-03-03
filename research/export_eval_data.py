#!/usr/bin/env python3
# Export NFCorpus and SciFact eval data to DSPy-friendly format.
# Output: research/artifacts/{dataset}_train.jsonl, {dataset}_dev.jsonl
#
# Usage:
#   pip install beir
#   python research/export_eval_data.py

import json
import random
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Seed for reproducibility
random.seed(42)


def load_beir_dataset(name: str):
    """Load a BEIR dataset, downloading if needed."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    data_path = Path(f"/tmp/beir/{name}")
    if not data_path.exists():
        out_dir = util.download_and_unzip(url, str(data_path.parent))
        data_path = Path(out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    return corpus, queries, qrels


def export_reranking_pairs(dataset: str, corpus, queries, qrels, max_pairs: int = 1000):
    """Export (query, doc, relevant: bool) triples for reranker training."""
    pairs = []
    for qid, rels in qrels.items():
        query = queries[qid]
        pos_ids = [did for did, score in rels.items() if score > 0]
        # Sample equal negatives from corpus (not in qrels for this query)
        neg_candidates = [did for did in corpus if did not in rels]
        neg_ids = random.sample(neg_candidates, min(len(pos_ids) * 3, len(neg_candidates), 10))

        for did in pos_ids[:3]:
            doc = corpus[did]
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            pairs.append({"query": query, "document": text[:2000], "relevant": True})

        for did in neg_ids:
            doc = corpus[did]
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            pairs.append({"query": query, "document": text[:2000], "relevant": False})

    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    split = int(len(pairs) * 0.8)
    return pairs[:split], pairs[split:]


def export_expansion_queries(dataset: str, queries, qrels, max_queries: int = 200):
    """Export queries with qrel doc IDs for expansion recall evaluation."""
    items = []
    for qid, rels in qrels.items():
        pos_ids = [did for did, score in rels.items() if score > 0]
        if pos_ids:
            items.append({
                "query": queries[qid],
                "relevant_doc_ids": pos_ids,
                "qid": qid,
            })

    random.shuffle(items)
    items = items[:max_queries]
    split = int(len(items) * 0.8)
    return items[:split], items[split:]


def write_jsonl(path: Path, records: list):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(records)} records → {path}")


def main():
    for dataset in ["nfcorpus", "scifact"]:
        print(f"\nLoading {dataset}...")
        corpus, queries, qrels = load_beir_dataset(dataset)
        print(f"  corpus={len(corpus)}, queries={len(queries)}, qrels={len(qrels)}")

        train, dev = export_reranking_pairs(dataset, corpus, queries, qrels)
        write_jsonl(ARTIFACTS / f"{dataset}_rerank_train.jsonl", train)
        write_jsonl(ARTIFACTS / f"{dataset}_rerank_dev.jsonl", dev)

        train_q, dev_q = export_expansion_queries(dataset, queries, qrels)
        write_jsonl(ARTIFACTS / f"{dataset}_expand_train.jsonl", train_q)
        write_jsonl(ARTIFACTS / f"{dataset}_expand_dev.jsonl", dev_q)

    print("\nDone. Run research/dspy_optimize.py next.")


if __name__ == "__main__":
    main()
