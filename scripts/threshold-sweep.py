#!/usr/bin/env python3
"""
threshold-sweep.py — Grid-search strong-signal thresholds from per-query signal data.

Usage:
  python3 scripts/threshold-sweep.py <signals-dir> [signals-dir ...]

Each signals-dir must contain bm25.jsonl, vector.jsonl, hybrid.jsonl produced by:
  beir-eval.py run --signals --signals-output <signals-dir>

Output: threshold sweep tables for tier-0 (BM25) and tier-1 (fused) gates,
plus coverage gating analysis across corpus sizes.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def load_qrels(signals_dir: Path) -> dict:
    """Load qrels from test-data/{dataset}/qrels/test.tsv adjacent to signals dir.
    signals_dir is e.g. logs/signals/fiqa or logs/signals/miracl-ko-s1000-p1.
    Strips sample suffix to find the base dataset qrels.
    """
    # Strip sample suffix: miracl-ko-s1000-p1 -> miracl-ko, fiqa -> fiqa
    import re
    base = re.sub(r"-s\d+-p\d+$", "", signals_dir.name)
    # Resolve relative to repo root (signals_dir may be relative)
    repo_root = Path(__file__).parent.parent
    qrels_path = repo_root / "test-data" / base / "qrels" / "test.tsv"
    if not qrels_path.exists():
        print(f"  WARNING: qrels not found at {qrels_path}", file=sys.stderr)
        return {}
    qrels: dict = {}
    with open(qrels_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
            qrels.setdefault(qid, {})[did] = score
    return qrels


# ── Load per-query data ──────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_signals_dir(signals_dir: Path) -> dict | None:
    """Load bm25+vector+hybrid JSONL from a signals directory. Returns None if incomplete."""
    bm25_path = signals_dir / "bm25.jsonl"
    vec_path = signals_dir / "vector.jsonl"
    hyb_path = signals_dir / "hybrid.jsonl"
    if not (bm25_path.exists() and vec_path.exists() and hyb_path.exists()):
        missing = [p.name for p in [bm25_path, vec_path, hyb_path] if not p.exists()]
        print(f"  WARNING: {signals_dir}: missing {', '.join(missing)} — skipping", file=sys.stderr)
        return None

    bm25_rows = {r["query_id"]: r for r in load_jsonl(bm25_path)}
    vec_rows = {r["query_id"]: r for r in load_jsonl(vec_path)}
    hyb_rows = {r["query_id"]: r for r in load_jsonl(hyb_path)}

    # Intersect on query_id
    query_ids = set(bm25_rows) & set(vec_rows) & set(hyb_rows)
    if len(query_ids) < len(bm25_rows) * 0.9:
        print(f"  WARNING: {signals_dir}: only {len(query_ids)}/{len(bm25_rows)} queries complete")

    qrels = load_qrels(signals_dir)

    queries = []
    for qid in sorted(query_ids):
        b = bm25_rows[qid]
        v = vec_rows[qid]
        h = hyb_rows[qid]
        rel = qrels.get(qid, {})
        fused_ndcg = offline_fused_ndcg(
            b.get("ranked", []), v.get("ranked", []), rel if rel else None
        )
        queries.append({
            "query_id": qid,
            "bm25_top": b.get("bm25_top", 0.0),
            "bm25_gap": b.get("bm25_gap", 0.0),
            "fused_top": h.get("fused_top", 0.0),
            "fused_gap": h.get("fused_gap", 0.0),
            "ndcg10_bm25": b.get("ndcg10", 0.0),
            "ndcg10_vector": v.get("ndcg10", 0.0),
            "ndcg10_hybrid": h.get("ndcg10", 0.0),
            "ndcg10_fused": fused_ndcg,  # offline fusion of bm25+vector ranked lists
        })

    missing_qrels = sum(1 for q in queries if not qrels.get(q["query_id"]))
    if missing_qrels:
        print(f"  WARNING: {missing_qrels} queries have no qrels (ndcg10_fused=0 for those)")

    return {
        "queries": queries,
        "label": signals_dir.name,
        "n": len(queries),
    }


# ── Offline fused nDCG ───────────────────────────────────────────────────────

VEC_ALPHA = 0.80


def offline_fused_ndcg(ranked_bm25: list, ranked_vector: list,
                        relevant: dict | None, k: int = 10) -> float:
    """Compute nDCG@k from offline score-fusion of BM25 and vector ranked lists.
    Mirrors the Weighted plan of score_fusion_two_list in hybrid.rs: score = 0.80*vec + 0.20*bm25.
    The Keyword and CrossScript plans (ADR-0002) are not simulated here.
    Uses rank-based score proxy since we don't have raw scores in ranked lists.
    """
    if relevant is None:
        return 0.0
    # Assign scores: position-based decay (1/(rank+1)) as proxy for normalized scores.
    # This is approximate — actual fusion uses normalized BM25 and cosine scores.
    # For threshold analysis, relative ordering is what matters.
    scores = {}
    for rank, doc_id in enumerate(ranked_bm25):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - VEC_ALPHA) * (1.0 / (rank + 1))
    for rank, doc_id in enumerate(ranked_vector):
        scores[doc_id] = scores.get(doc_id, 0.0) + VEC_ALPHA * (1.0 / (rank + 1))

    ranked_fused = sorted(scores, key=lambda d: scores[d], reverse=True)

    # Compute nDCG@k
    idcg = 0.0
    for i, rel in enumerate(sorted(relevant.values(), reverse=True)[:k]):
        idcg += (2 ** rel - 1) / math.log2(i + 2)
    if idcg == 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(ranked_fused[:k]):
        rel = relevant.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg / idcg


# ── Tier-0 BM25 sweep ────────────────────────────────────────────────────────

BM25_FLOORS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
BM25_GAPS   = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
HARM_EPSILON = 0.01  # nDCG loss threshold to count as "harmed"


def sweep_bm25(queries: list) -> list:
    """Grid search over (floor, gap) for tier-0 BM25 shortcut."""
    n = len(queries)
    results = []
    for floor in BM25_FLOORS:
        for gap in BM25_GAPS:
            fires = [q for q in queries if q["bm25_top"] >= floor and q["bm25_gap"] >= gap]
            harmed = [q for q in fires if q["ndcg10_hybrid"] - q["ndcg10_bm25"] > HARM_EPSILON]
            loss_values = [q["ndcg10_hybrid"] - q["ndcg10_bm25"] for q in fires]
            results.append({
                "floor": floor,
                "gap": gap,
                "fire_n": len(fires),
                "fire_rate": len(fires) / n if n else 0,
                "harm_n": len(harmed),
                "harm_rate": len(harmed) / len(fires) if fires else 0,
                "mean_loss": sum(loss_values) / len(loss_values) if loss_values else 0,
            })
    return results


# ── Tier-1 fused sweep ───────────────────────────────────────────────────────

FUSED_PRODUCTS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]
# Floor dimension removed — empirically irrelevant: all fused_top scores exceed any
# reasonable floor threshold, so product alone drives the fire/harm tradeoff.


def sweep_fused(queries: list) -> list:
    """Sweep product threshold for tier-1 fused shortcut.
    Harm = full pipeline nDCG - offline fused nDCG > epsilon.
    """
    n = len(queries)
    results = []
    for product in FUSED_PRODUCTS:
        fires = [
            q for q in queries
            if q["fused_top"] * q["fused_gap"] >= product
        ]
        harmed = [
            q for q in fires
            if q["ndcg10_hybrid"] - q["ndcg10_fused"] > HARM_EPSILON
        ]
        losses = [q["ndcg10_hybrid"] - q["ndcg10_fused"] for q in fires]
        results.append({
            "product": product,
            "fire_n": len(fires),
            "fire_rate": len(fires) / n if n else 0,
            "harm_n": len(harmed),
            "harm_rate": len(harmed) / len(fires) if fires else 0,
            "mean_loss": sum(losses) / len(losses) if losses else 0,
        })
    return results


# ── Print tables ─────────────────────────────────────────────────────────────

def print_bm25_table(sweep: list, label: str, n: int, current=(0.75, 0.10)):
    print(f"\nTier-0 BM25 Threshold Sweep — {label} ({n} queries)")
    print("=" * 72)
    print(f"  {'floor':>5}  {'gap':>5}  {'fire%':>6}  {'harm%':>6}  {'avg_loss':>9}  {'note'}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*20}")
    for r in sorted(sweep, key=lambda x: (x["floor"], x["gap"])):
        note = ""
        if (r["floor"], r["gap"]) == current:
            note = "<-- CURRENT"
        if r["harm_rate"] < 0.05 and r["fire_rate"] > 0.10:
            note += " CANDIDATE" if not note else " *"
        print(f"  {r['floor']:>5.2f}  {r['gap']:>5.2f}  "
              f"{r['fire_rate']*100:>5.1f}%  {r['harm_rate']*100:>5.1f}%  "
              f"{r['mean_loss']:>9.4f}  {note}")
    print()


def print_fused_table(sweep: list, label: str, n: int, current=0.06):
    print(f"\nTier-1 Fused Threshold Sweep — {label} ({n} queries)")
    print("=" * 65)
    print(f"  {'product':>7}  {'fire%':>6}  {'harm%':>6}  {'avg_loss':>9}  note")
    print(f"  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*20}")
    for r in sorted(sweep, key=lambda x: x["product"]):
        note = ""
        if r["product"] == current:
            note = "<-- CURRENT"
        if r["harm_rate"] < 0.05 and r["fire_rate"] > 0.10:
            note += " CANDIDATE" if not note else " *"
        print(f"  {r['product']:>7.3f}  "
              f"{r['fire_rate']*100:>5.1f}%  {r['harm_rate']*100:>5.1f}%  "
              f"{r.get('mean_loss', 0):>9.4f}  {note}")
    print()


def print_coverage_summary(datasets: list):
    """Show how optimal BM25 threshold shifts across corpus sizes."""
    print("\nCoverage Gating Analysis — BM25 floor threshold vs corpus size")
    print("=" * 60)
    print(f"  {'label':<30}  {'queries':>7}  {'best_floor':>10}  {'fire%':>6}  {'harm%':>6}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*10}  {'-'*6}  {'-'*6}")
    for d in datasets:
        sweep = d.get("bm25_sweep", [])
        if not sweep:
            continue
        # Best floor at harm_rate < 5% with lowest floor (most aggressive)
        candidates = [r for r in sweep if r["harm_rate"] < 0.05 and r["gap"] == 0.10]
        if candidates:
            best = min(candidates, key=lambda r: r["floor"])
            print(f"  {d['label']:<30}  {d['n']:>7}  {best['floor']:>10.2f}  "
                  f"{best['fire_rate']*100:>5.1f}%  {best['harm_rate']*100:>5.1f}%")
        else:
            print(f"  {d['label']:<30}  {d['n']:>7}  {'n/a':>10}")
    print()
    print("If best_floor is stable across corpus sizes: no coverage-dependent gating needed.")
    print("If best_floor decreases as size grows: consider scaling threshold with corpus size.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sweep strong-signal thresholds from per-query signal data")
    ap.add_argument("signals_dirs", nargs="+", metavar="DIR",
                    help="Directories containing bm25.jsonl, vector.jsonl, hybrid.jsonl")
    ap.add_argument("--output", "-o", help="Write JSON sweep results to file")
    args = ap.parse_args()

    datasets = []
    for d in args.signals_dirs:
        path = Path(d)
        print(f"Loading {path}...")
        data = load_signals_dir(path)
        if data is None:
            continue
        queries = data["queries"]
        print(f"  {len(queries)} queries loaded")

        bm25_sweep = sweep_bm25(queries)
        fused_sweep = sweep_fused(queries)
        data["bm25_sweep"] = bm25_sweep
        data["fused_sweep"] = fused_sweep
        datasets.append(data)

        print_bm25_table(bm25_sweep, data["label"], len(queries))
        print_fused_table(fused_sweep, data["label"], len(queries))

    if len(datasets) > 1:
        print_coverage_summary(datasets)

    if args.output:
        out = []
        for d in datasets:
            out.append({
                "label": d["label"],
                "n": d["n"],
                "bm25_sweep": d["bm25_sweep"],
                "fused_sweep": d["fused_sweep"],
            })
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSweep results written to {args.output}")


if __name__ == "__main__":
    main()
