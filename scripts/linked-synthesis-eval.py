#!/usr/bin/env python3
"""Evaluate linked retrieval as synthesis evidence assembly.

This is intentionally model-free. It verifies that a mixed code/Markdown
collection returns the code hit plus explicit related knowledge needed to answer
the question. The expected synthesis text is a human-readable contract; the
script checks evidence coverage rather than asking an LLM to generate prose.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def flatten_result_text(result: dict) -> str:
    parts: list[str] = []
    for key in ("path", "title", "snippet", "content", "symbol"):
        value = result.get(key)
        if value:
            parts.append(str(value))
    for marker in result.get("markers") or []:
        parts.append(str(marker))
    for item in result.get("related") or []:
        for key in ("path", "title", "symbol", "snippet", "target", "raw"):
            value = item.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    text = re.sub(r"</?b>", "", text)
    return re.sub(r"\s+", " ", text).lower()


def evaluate_task(results: list[dict], task: dict) -> list[str]:
    failures: list[str] = []
    if not results:
        return ["no search results"]

    expected_primary = task["expected_primary_path"]
    primary = next((r for r in results if r.get("path") == expected_primary), results[0])
    if primary.get("path") != expected_primary:
        failures.append(
            f"expected primary path {expected_primary!r}, got {primary.get('path')!r}"
        )

    markers = set(primary.get("markers") or [])
    for marker in task.get("required_markers", []):
        if marker not in markers:
            failures.append(f"missing marker {marker!r} on primary result")

    related_paths = {item.get("path") for item in primary.get("related") or []}
    for path in task.get("required_related_paths", []):
        if path not in related_paths:
            failures.append(f"missing related path {path!r}")

    related_kinds = {item.get("kind") for item in primary.get("related") or []}
    for kind in task.get("required_related_kinds", []):
        if kind not in related_kinds:
            failures.append(f"missing related link kind {kind!r}")

    related_targets = {item.get("target") for item in primary.get("related") or []}
    for target in task.get("required_related_targets", []):
        if target not in related_targets:
            failures.append(f"missing related link target {target!r}")

    evidence = normalize_text(flatten_result_text(primary))
    for term in task.get("required_evidence_terms", []):
        if normalize_text(term) not in evidence:
            failures.append(f"missing evidence term {term!r}")

    synthesis = task.get("expected_synthesis", "").lower()
    for term in task.get("required_evidence_terms", []):
        first_word = term.lower().split()[0]
        if first_word and first_word not in synthesis:
            failures.append(
                f"expected_synthesis does not mention evidence anchor {first_word!r}"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-bin", default="target/debug/ir")
    parser.add_argument("--fixture", default="test-data/fixtures/linked-synthesis")
    parser.add_argument("--keep-state", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    fixture = (repo / args.fixture).resolve()
    ir_bin = (repo / args.ir_bin).resolve()
    evals = json.loads((fixture / "evals.json").read_text())
    collection = evals.get("collection", "linked-synthesis")

    state = Path(tempfile.mkdtemp(prefix="ir-linked-synthesis-"))
    env = os.environ.copy()
    env["IR_CONFIG_DIR"] = str(state / "ir")
    env["IR_DISABLE_SHORTCUTS"] = "1"
    try:
        run(
            [str(ir_bin), "collection", "add", collection, str(fixture), "--preset", "mixed"],
            env=env,
            cwd=repo,
        )
        run([str(ir_bin), "update", collection], env=env, cwd=repo)

        failures: dict[str, list[str]] = {}
        for task in evals["tasks"]:
            stdout = run(
                [
                    str(ir_bin),
                    "search",
                    task["query"],
                    "--mode",
                    "bm25",
                    "-c",
                    collection,
                    "-n",
                    "5",
                    "--chunk",
                    "--related",
                    "5",
                    "--json",
                    "--quiet",
                ],
                env=env,
                cwd=repo,
            )
            results = json.loads(stdout)
            task_failures = evaluate_task(results, task)
            if task_failures:
                failures[task["id"]] = task_failures

        if failures:
            print(json.dumps({"ok": False, "failures": failures}, indent=2), file=sys.stderr)
            return 1

        print(
            json.dumps(
                {
                    "ok": True,
                    "fixture": str(fixture.relative_to(repo)),
                    "tasks": len(evals["tasks"]),
                },
                indent=2,
            )
        )
        return 0
    finally:
        if args.keep_state:
            print(f"state kept at {state}", file=sys.stderr)
        else:
            shutil.rmtree(state, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
