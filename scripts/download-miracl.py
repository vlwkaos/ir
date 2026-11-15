#!/usr/bin/env python3
"""Download a MIRACL language (dev split) in BEIR format. Stdlib only.

Usage: python3 scripts/download-miracl.py <lang> <out-dir>
Output: <out-dir>/{corpus.jsonl, queries.jsonl, qrels/test.tsv}

Reads the raw files of the HF repos at pinned revisions and writes what their loader
scripts (miracl.py, miracl-corpus.py) yield: corpus in sorted docs-N.jsonl.gz order,
dev topics in file order, positives = qrels with rel 1 whose doc is in the corpus.
^ `datasets` 4 no longer runs loader scripts, so load_dataset("miracl/miracl") fails.
"""

import gzip
import io
import json
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets"
CORPUS_REV = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"  # miracl/miracl-corpus
TOPICS_REV = "5be20db9509754dadad47689368639fcec739c00"  # miracl/miracl
SHARDS = {"ko": 3, "zh": 10}  # ^ docs-N.jsonl.gz per language at CORPUS_REV


def fetch(url: str):
    return urllib.request.urlopen(url, timeout=120)


def lines(url: str):
    with fetch(url) as r:
        yield from io.TextIOWrapper(r, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in SHARDS:
        sys.exit(f"usage: download-miracl.py <{'|'.join(SHARDS)}> <out-dir>")
    lang, out = sys.argv[1], Path(sys.argv[2])
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    corpus_path = out / "corpus.jsonl"
    part = corpus_path.with_suffix(".jsonl.part")
    docids = set()
    if corpus_path.exists():
        print(f"corpus already exists ({corpus_path.stat().st_size // 1_000_000} MB), reading ids")
        with open(corpus_path, encoding="utf-8") as f:
            docids = {json.loads(l)["_id"] for l in f}
    else:
        with open(part, "w", encoding="utf-8") as f:
            for i in sorted(range(SHARDS[lang]), key=str):  # ^ loader sorts paths as strings
                url = f"{HF}/miracl/miracl-corpus/resolve/{CORPUS_REV}/miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz"
                print(f"corpus shard {i}: {url}", flush=True)
                with fetch(url) as r, gzip.open(r, "rt", encoding="utf-8") as g:
                    for line in g:
                        d = json.loads(line)
                        docids.add(d["docid"])
                        f.write(json.dumps({"_id": d["docid"], "title": d.get("title", ""), "text": d["text"]},
                                           ensure_ascii=False) + "\n")
        part.rename(corpus_path)
        print(f"  {len(docids)} passages ({corpus_path.stat().st_size // 1_000_000} MB)")

    base = f"{HF}/miracl/miracl/resolve/{TOPICS_REV}/miracl-v1.0-{lang}"
    topics = {}
    for line in lines(f"{base}/topics/topics.miracl-v1.0-{lang}-dev.tsv"):
        qid, text = line.strip().split("\t")
        topics[qid] = text
    qrels = {}
    for line in lines(f"{base}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv"):
        qid, _, docid, rel = line.strip().split("\t")
        qrels.setdefault(qid, {})[docid] = int(rel)
    with open(out / "queries.jsonl", "w", encoding="utf-8") as qf, \
            open(out / "qrels" / "test.tsv", "w", encoding="utf-8") as rf:
        rf.write("query-id\tcorpus-id\tscore\n")
        for qid, text in topics.items():
            qf.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
            for docid, rel in qrels.get(qid, {}).items():
                if rel == 1 and docid in docids:
                    rf.write(f"{qid}\t{docid}\t1\n")
    print(f"  {len(topics)} queries; dataset at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
