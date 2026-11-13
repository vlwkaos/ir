#!/usr/bin/env python3
"""Kaggle kernel driver: run scripts/bench.sh on attached read-only datasets.

See research/kaggle-eval.md. Stdlib only. The workflow prepends an
os.environ.update({...}) line with IRK_DATASET, IRK_MODE, IRK_GIT7 (the git7 the
bundle must have: a dataset version that is still processing mounts the old one),
IRK_SIZE (bench.sh --size, seed 42) and IRK_PREPROCESSOR (alias, e.g. ko).

Inputs, found by marker files anywhere under IRK_INPUT (default /kaggle/input):
  bundle   bundle.json + ir + bench scripts + libgomp.so.* (scripts/kaggle/bundle.sh)
  corpus   DATASET file naming the dataset, corpus.jsonl, queries.jsonl, test.tsv
  models   <embedding_file>, <reranker_file>, <expander_file> from bundle.json
           (expander stays off by the retrieval profile, but bench.sh names it and
           ir resolves every IR_*_MODEL before starting the daemon)
  lindera  PREPROCESSOR (alias=, lindera=<tag>, dict=<name>) + lindera binary + dictionary
           files, flat (only with IRK_PREPROCESSOR)
  index    INDEX_KEY + eval-<label>.sqlite[.usearch]  (optional; label = bench.sh
           DATASET_LABEL, <dataset>-s<size>-p42 with IRK_SIZE)

Outputs in IRK_WORKING (default /kaggle/working):
  result/<label>/<git7>.json, bench.log, env.txt, daemon.log
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
SIZE = os.environ.get("IRK_SIZE", "")  # ^ bench.sh --size (seed 42); empty: full corpus
PREPROCESSOR = os.environ.get("IRK_PREPROCESSOR", "")  # ^ bench.sh's alias for DATASET (ko, zh)
LABEL = f"{DATASET}-s{SIZE}-p42" if SIZE else DATASET  # ^ bench.sh DATASET_LABEL
# ^ CLAUDE.md sentinel probes (input, output of the installed command). The output is the macOS
#   lindera v3.0.5 one, so a match shows the Linux binary tokenizes alike. ko gives 2 lines for
#   3: its stop-tag filter drops ".", the 0-output case the IRSENTINEL protocol covers
#   (preprocess.rs; tests/e2e_index.rs preprocessor_that_drops_a_line_does_not_stall_the_update).
PROBES = {"ko": (".\n안녕하세요\ntest\n", "안녕 세요\ntest\n"),
          "zh": ("。\n你好世界\ntest\n", "。\n你好 世界\ntest\n")}
EXPECT_GIT7 = os.environ.get("IRK_GIT7", "")
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


def marker(path: Path) -> "dict[str, str]":
    """key=value lines of a marker file (PREPROCESSOR, written by ci.sh)."""
    return dict(l.split("=", 1) for l in path.read_text().splitlines() if "=" in l)


def install_preprocessor(src: Path, ir: Path, env: "dict[str, str]") -> str:
    """Install the attached lindera + dictionary as `ir preprocessor install` would, offline.

    Files go where the install puts them under HOME/.config/ir (bench-env.sh copies that
    config.yml and links its preprocessors/). The install then finds them and skips the
    download (install_lindera_binary, install_lindera_dict in src/main.rs), so config.yml
    gets ir's own $IR_DIR command. Returns the index key lines for this preprocessor.
    """
    m = marker(src / "PREPROCESSOR")
    ir_dir = Path(env["HOME"]) / ".config/ir"
    pre = ir_dir / "preprocessors"
    (pre / f"lindera-{m['lindera']}").mkdir(parents=True)
    (pre / m["dict"]).mkdir()
    shutil.copy(src / "lindera", pre / f"lindera-{m['lindera']}" / "lindera")
    (pre / f"lindera-{m['lindera']}" / "lindera").chmod(0o755)
    dict_files = sorted(f for f in src.iterdir()
                        if f.is_file() and f.name not in ("lindera", "PREPROCESSOR", "dataset-metadata.json"))
    for f in dict_files:
        shutil.copy(f, pre / m["dict"] / f.name)
    r = subprocess.run([str(ir), "preprocessor", "install", PREPROCESSOR], env=env,
                       capture_output=True, text=True)
    print(r.stdout + r.stderr, flush=True)
    cmd = next((l.split("→", 1)[1].strip() for l in r.stdout.splitlines() if "→" in l), "")
    if r.returncode != 0 or not cmd:
        sys.exit(f"[irk] ir preprocessor install {PREPROCESSOR} failed (exit {r.returncode})")
    if "download" in r.stdout:
        sys.exit("[irk] the install tried to download: the attached files are not where ir looks")
    # ^ ir splits the command on whitespace and expands $IR_DIR per arg (preprocess.rs spawn).
    argv = [a.replace("$IR_DIR", str(ir_dir)) for a in cmd.split()]
    probe, want = PROBES[PREPROCESSOR]
    p = subprocess.run(argv, input=probe, capture_output=True, text=True, timeout=120)
    log(f"sentinel probe {PREPROCESSOR}: {len(p.stdout.splitlines())} lines out for "
        f"{len(probe.splitlines())} in: {p.stdout!r}")
    if p.returncode != 0 or p.stdout != want:
        sys.exit(f"[irk] sentinel probe: expected {want!r} (exit {p.returncode}: {p.stderr.strip()}). "
                 "A lindera or flag change needs the CLAUDE.md probe review")
    dict_sha = hashlib.sha256("".join(f"{f.name}={sha256(f)}\n" for f in dict_files).encode()).hexdigest()
    return (f"preprocessor={cmd}\nlindera_sha256={sha256(src / 'lindera')}\n"
            f"{m['dict']}_sha256={dict_sha}\n")


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return f"$ {cmd}\n{r.stdout}{r.stderr}"


def main() -> int:
    if not DATASET:
        sys.exit("[irk] IRK_DATASET is not set")
    if PREPROCESSOR and PREPROCESSOR not in PROBES:
        sys.exit(f"[irk] no sentinel probe for preprocessor {PREPROCESSOR!r}")
    WORKING.mkdir(parents=True, exist_ok=True)

    bundle = one_dir("bundle.json", "bundle")
    if bundle is None:
        sys.exit(f"[irk] no bundle (bundle.json) under {INPUT}")
    meta = json.loads((bundle / "bundle.json").read_text())
    git7 = meta["git7"]
    if EXPECT_GIT7 and git7 != EXPECT_GIT7:
        sys.exit(f"[irk] stale bundle: attached {git7}, expected {EXPECT_GIT7}")
    corpus = one_dir("DATASET", "corpus", lambda d: (d / "DATASET").read_text().strip() == DATASET)
    if corpus is None:
        sys.exit(f"[irk] no corpus dataset with DATASET={DATASET} under {INPUT}")
    models = {}
    for kind in ("embedding", "reranker", "expander"):
        d = one_dir(meta[f"{kind}_file"], f"{kind} model")
        if d is None:
            sys.exit(f"[irk] model {meta[f'{kind}_file']} not attached")
        models[kind] = d / meta[f"{kind}_file"]
    lindera = None
    if PREPROCESSOR:
        lindera = one_dir("PREPROCESSOR", f"{PREPROCESSOR} preprocessor",
                          lambda d: marker(d / "PREPROCESSOR").get("alias") == PREPROCESSOR)
        if lindera is None:
            sys.exit(f"[irk] no preprocessor dataset with alias={PREPROCESSOR} under {INPUT}")
    collection = f"eval-{LABEL}"
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
    # ^ The image may lack libgomp (llama-openmp links it dynamically): bundle.sh ships the
    #   build host's copy. Flat copy, since Kaggle datasets skip directories on upload.
    lib = ROOT / "lib"
    lib.mkdir()
    for so in bundle.glob("lib*.so*"):
        shutil.copy(so, lib / so.name)
    # ^ Run 43120474122: kernels pushed through the API got no GPU on this account (script or
    #   notebook, internet on or off, machine_shape set or not). Every mode then runs the same
    #   CUDA build on the CPU backend. ir lists libcuda.so.1 as NEEDED and cannot start without
    #   it, and ggml-cuda takes the address of cu* driver functions, so the loader wants each one
    #   (run 43121732580: undefined symbol cuMemCreate). Build a stub that defines every cu*
    #   symbol ir imports as "CUDA_ERROR_NO_DEVICE"; ggml-cuda then reports no device.
    gpu = any(Path("/dev").glob("nvidia[0-9]*")) or \
        subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0
    cpu = not gpu and sys.platform.startswith("linux")  # ^ not on a macOS dry run
    if gpu:
        log("backend: GPU")
    elif cpu:
        # ^ build.sh compiles for x86-64-v3; without these flags ir dies on SIGILL.
        flags = set(next((l.split(":", 1)[1].split() for l in open("/proc/cpuinfo") if l.startswith("flags")), []))
        missing = {"avx2", "fma", "f16c", "bmi2"} - flags
        if missing:
            sys.exit(f"[irk] no GPU and the CPU lacks {sorted(missing)} (bundle is built for x86-64-v3)")
        log(f"backend: CPU, no GPU in this session ({os.cpu_count()} CPUs)")
        if subprocess.run("ldconfig -p | grep -q libcuda.so.1", shell=True).returncode != 0:
            nm = subprocess.run(["nm", "-D", "--undefined-only", str(ir)], capture_output=True, text=True)
            syms = sorted({f.split("@")[0] for l in nm.stdout.splitlines() for f in l.split()[-1:]
                           if f.startswith("cu") and f[2:3].isupper()})
            (lib / "stub.c").write_text("".join(f"int {n}(void) {{ return 100; }}\n" for n in syms))
            stub = subprocess.run(["gcc", "-shared", "-o", str(lib / "libcuda.so.1"), "-Wl,-soname,libcuda.so.1",
                                   str(lib / "stub.c")], capture_output=True, text=True)
            log(f"no libcuda.so.1 in this session; using a stub of {len(syms)} cu* symbols"
                if stub.returncode == 0 and syms else f"stub failed: nm={nm.stderr.strip()} gcc={stub.stderr.strip()}")
    ld_path = ":".join(p for p in (str(lib), os.environ.get("LD_LIBRARY_PATH", "")) if p)
    data = ROOT / "data" / DATASET
    (data / "qrels").mkdir(parents=True)
    for src, dst in (("corpus.jsonl", "corpus.jsonl"), ("queries.jsonl", "queries.jsonl"),
                     ("test.tsv", "qrels/test.tsv")):
        if not (corpus / src).is_file():
            sys.exit(f"[irk] corpus dataset lacks {src}")
        (data / dst).symlink_to(corpus / src)

    os.environ["LD_LIBRARY_PATH"] = ld_path
    env_txt = "\n".join(sh(c) for c in (
        "cat /etc/os-release", "ldd --version | head -1", "which gcc cc", "nvidia-smi", f"ldd {ir}",
        # ^ Run 43117637669: the session had no GPU. Show what the container does mount.
        "ls -la /dev | grep -i nvidia", "cat /proc/driver/nvidia/version", "ls /usr/local/nvidia/lib64",
        "env | grep -i -E 'nvidia|cuda'", "ldconfig -p | grep -i -E 'libcuda|nvidia'", "nproc; free -g; df -h /kaggle/working /tmp",
        "lscpu", f"objdump -d --no-show-raw-insn {ir} | grep -c vfmadd",  # ^ >0: built for x86-64-v3
        f"{ir} --version", f"ls -laR {INPUT}"))
    (WORKING / "env.txt").write_text(env_txt)
    if "=> not found" in env_txt:
        print(env_txt)
        if "libcuda.so.1 => not found" in env_txt:
            print("[irk] no NVIDIA driver in this session: the kernel was not given a GPU")
        sys.exit("[irk] ir has unresolved shared libraries (see env.txt)")

    env = {k: v for k, v in os.environ.items()
           if k not in ("XDG_CONFIG_HOME", "IR_CONFIG_DIR", "IR_CONFIG_FILE", "IR_MODEL_DIRS")}
    env.update({
        "HOME": str(ROOT / "home"),  # ^ no ambient ~/.config/ir to copy in
        "HF_HUB_OFFLINE": "1",
        "IR_BENCH_GUARD": "0",  # ^ only matters for a local dry run on macOS
    })
    if cpu:
        # ^ Pin models to the CPU backend. A tier-2 query on 4 vCPU can take minutes: raise the
        #   daemon reply limit (120 s, else ir falls back to BM25 without a word under -q) and
        #   beir-eval.py's 60 s per query (else it scores an empty ranking).
        env.update({"IR_FORCE_CPU_BACKEND": "1", "IR_GPU_LAYERS": "0",
                    "IR_DAEMON_REPLY_TIMEOUT": "1800", "IR_BENCH_QUERY_TIMEOUT": "1900"})
    env.update({f"IR_{kind.upper()}_MODEL": str(path) for kind, path in models.items()})
    log(f"hashing {models['embedding'].name}")
    key = meta["index_key_src"] + f"dataset={LABEL}\nembedding_sha256={sha256(models['embedding'])}\n"
    if lindera is not None:
        key += install_preprocessor(lindera, ir, env)
    args = ["bash", str(repo / "scripts/bench.sh"), DATASET, "--mode", MODE,
            "--ir-bin", str(ir), "--git7", git7, "--data-dir", str(data)]
    if SIZE:
        args += ["--size", SIZE, "--seed", "42"]
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
    result = repo / "logs/results" / LABEL / f"{git7}.json"
    if result.is_file():
        (WORKING / "result" / LABEL).mkdir(parents=True, exist_ok=True)
        shutil.copy(result, WORKING / "result" / LABEL / result.name)
    if status == 0 and out_index.is_dir():
        (out_index / "INDEX_KEY").write_text(key)
        log(f"index: saved to {out_index} (upload as a new index dataset version)")
    log(f"bench.sh exit {status}")
    return status


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as e:
        code = e.code
    # ^ Raise instead of sys.exit: a notebook cell that exits may still end the run as complete.
    if code not in (0, None):
        raise RuntimeError(f"[irk] failed: {code}")
