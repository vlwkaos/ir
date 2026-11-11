#!/usr/bin/env python3
"""Kaggle kernel driver: run scripts/bench.sh on attached read-only datasets.

See research/kaggle-eval.md. Stdlib only. The workflow prepends an
os.environ.update({...}) line with IRK_DATASET and IRK_MODE.

Inputs, found by marker files anywhere under IRK_INPUT (default /kaggle/input):
  bundle   bundle.json + ir + bench scripts      (scripts/kaggle/bundle.sh)
  corpus   DATASET file naming the dataset, corpus.jsonl, queries.jsonl, test.tsv
  models   <embedding_file>, <reranker_file>, <expander_file> from bundle.json
           (expander stays off by the retrieval profile, but bench.sh names it and
           ir resolves every IR_*_MODEL before starting the daemon)
  index    INDEX_KEY + eval-<dataset>.sqlite[.usearch]  (optional)

Outputs in IRK_WORKING (default /kaggle/working):
  result/<dataset>/<git7>.json, bench.log, env.txt, daemon.log
  index/  only when the index was (re)built: upload it as the index dataset
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

DATASET = os.environ.get("IRK_DATASET", "")
MODE = os.environ.get("IRK_MODE", "all")
INPUT = Path(os.environ.get("IRK_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("IRK_WORKING", "/kaggle/working"))
# ^ Short root: the daemon socket path lives under it (sun_path limit ~104 bytes).
ROOT = Path(os.environ.get("IRK_ROOT", "/tmp/irk"))


def log(msg: str) -> None:
    print(f"[irk] {msg}", flush=True)


def find_dirs(marker: str) -> "list[Path]":
    """Dirs under INPUT that contain `marker` (follows the mount layout, whatever it is)."""
    return sorted({p.parent for p in INPUT.rglob(marker)})


def one_dir(marker: str, what: str, pred=lambda d: True) -> Optional[Path]:
    hits = [d for d in find_dirs(marker) if pred(d)]
    if len(hits) > 1:
        sys.exit(f"[irk] more than one {what} attached: {hits}")
    return hits[0] if hits else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return f"$ {cmd}\n{r.stdout}{r.stderr}"


def main() -> int:
    if not DATASET:
        sys.exit("[irk] IRK_DATASET is not set")
    WORKING.mkdir(parents=True, exist_ok=True)

    bundle = one_dir("bundle.json", "bundle")
    if bundle is None:
        sys.exit(f"[irk] no bundle (bundle.json) under {INPUT}")
    meta = json.loads((bundle / "bundle.json").read_text())
    git7 = meta["git7"]
    corpus = one_dir("DATASET", "corpus", lambda d: (d / "DATASET").read_text().strip() == DATASET)
    if corpus is None:
        sys.exit(f"[irk] no corpus dataset with DATASET={DATASET} under {INPUT}")
    models = {}
    for kind in ("embedding", "reranker", "expander"):
        d = one_dir(meta[f"{kind}_file"], f"{kind} model")
        if d is None:
            sys.exit(f"[irk] model {meta[f'{kind}_file']} not attached")
        models[kind] = d / meta[f"{kind}_file"]
    collection = f"eval-{DATASET}"
    index = one_dir("INDEX_KEY", "index", lambda d: (d / f"{collection}.sqlite").is_file())

    # Writable work tree: scripts (bench.sh writes logs/ and .bench-state/ under
    # its repo root), binary with exec bit, and the dataset layout bench.sh expects.
    if ROOT.exists():
        shutil.rmtree(ROOT)
    repo = ROOT / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("bench.sh", "bench-env.sh", "beir-eval.py"):
        shutil.copy(bundle / name, repo / "scripts" / name)
    ir = ROOT / "ir"
    shutil.copy(bundle / "ir", ir)
    ir.chmod(0o755)
    data = ROOT / "data" / DATASET
    (data / "qrels").mkdir(parents=True)
    for src, dst in (("corpus.jsonl", "corpus.jsonl"), ("queries.jsonl", "queries.jsonl"),
                     ("test.tsv", "qrels/test.tsv")):
        if not (corpus / src).is_file():
            sys.exit(f"[irk] corpus dataset lacks {src}")
        (data / dst).symlink_to(corpus / src)

    env_txt = "\n".join(sh(c) for c in (
        "cat /etc/os-release", "nvidia-smi", f"ldd {ir}", f"{ir} --version", f"ls -laR {INPUT}"))
    (WORKING / "env.txt").write_text(env_txt)
    if "=> not found" in env_txt:
        print(env_txt)
        sys.exit("[irk] ir has unresolved shared libraries (see env.txt)")

    log(f"hashing {models['embedding'].name}")
    key = meta["index_key_src"] + f"dataset={DATASET}\nembedding_sha256={sha256(models['embedding'])}\n"
    args = ["bash", str(repo / "scripts/bench.sh"), DATASET, "--mode", MODE,
            "--ir-bin", str(ir), "--git7", git7, "--data-dir", str(data)]
    out_index = WORKING / "index"
    if index is not None and (index / "INDEX_KEY").read_text() == key:
        log(f"index: key match, reusing {index}")
        args += ["--index-in", str(index)]
    else:
        if index is None:
            log("index: none attached, building")
        else:
            log("index: key mismatch, rebuilding. attached vs expected:")
            print((index / "INDEX_KEY").read_text() + "---\n" + key, flush=True)
        if MODE == "bm25":
            # ^ bm25 prepare skips embedding; saving that index would serve vector runs.
            log("index: bm25 only, not saving the index")
        else:
            args += ["--index-out", str(out_index)]

    env = {k: v for k, v in os.environ.items()
           if k not in ("XDG_CONFIG_HOME", "IR_CONFIG_DIR", "IR_CONFIG_FILE", "IR_MODEL_DIRS")}
    env.update({
        "HOME": str(ROOT / "home"),  # ^ no ambient ~/.config/ir to copy in
        "HF_HUB_OFFLINE": "1",
        "IR_BENCH_GUARD": "0",  # ^ only matters for a local dry run on macOS
    })
    env.update({f"IR_{kind.upper()}_MODEL": str(path) for kind, path in models.items()})
    log("running: " + " ".join(args))
    with open(WORKING / "bench.log", "w", buffering=1) as bench_log:
        p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            bench_log.write(line)
        status = p.wait()

    state = repo / ".bench-state/bench/xdg/ir"
    if (state / "daemon.log").is_file():
        shutil.copy(state / "daemon.log", WORKING / "daemon.log")
    result = repo / "logs/results" / DATASET / f"{git7}.json"
    if result.is_file():
        (WORKING / "result" / DATASET).mkdir(parents=True, exist_ok=True)
        shutil.copy(result, WORKING / "result" / DATASET / result.name)
    if status == 0 and out_index.is_dir():
        (out_index / "INDEX_KEY").write_text(key)
        log(f"index: saved to {out_index} (upload as a new index dataset version)")
    log(f"bench.sh exit {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
