#!/usr/bin/env python3
"""
run_eval.py: task eval for the AMADEUS templates.

Runs each case in cases.py through the real TUI on a fresh copy of workspace/,
scores what the model left behind, and reports how often each check passes.
Needs Ollama and the model; scoring itself is deterministic.

    python run_eval.py                            # the cases with no shell tool, 3 runs each
    python run_eval.py --allow-shell              # also the code-helper cases
    python run_eval.py --runs 1 --case expense-notes

Runs use --auto, so writes are approved without asking. With --allow-shell, that
includes run_command: whatever the model proposes runs unconfined, as you. Read
eval/README.md first.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from cases import CASES, WORKSPACE, harness_checks
from driver import AMADEUS, TUI_BIN, run_task

RESULTS = Path(__file__).resolve().parent / "results"


def preflight(model: str) -> None:
    url = os.environ.get("AMADEUS_OLLAMA_URL", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.load(r)["models"]]
    except OSError as e:
        sys.exit(f"error: can't reach Ollama at {url} ({e})")
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"error: model {model} isn't pulled. Run: ollama pull {model}")
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=AMADEUS / "tui", check=True)
    if not TUI_BIN.is_file():
        sys.exit(f"error: no TUI binary at {TUI_BIN}")


def run_once(name: str, case: dict, out_dir: Path, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="amadeus-eval-") as tmp:
        ws, home = Path(tmp) / "ws", Path(tmp) / "home"
        home.mkdir()
        shutil.copytree(WORKSPACE, ws)
        result = run_task(ws, home, case["template"], case["task"], timeout)
        checks = case["score"](ws, result["answer"], result["events"]) + harness_checks(name, ws, result)
        shutil.copytree(ws, out_dir / "workspace", ignore=shutil.ignore_patterns("__pycache__"))
        if result["log"]:
            shutil.copy(result["log"], out_dir / "run.jsonl")
    return {
        "case": name,
        "status": result["status"],
        "seconds": round(result["seconds"], 1),
        "checks": [asdict(c) for c in checks],
    }


def report(runs: list[dict]) -> None:
    by_case = defaultdict(list)
    for r in runs:
        by_case[r["case"]].append(r)

    widths = [22, 5, 8, 8, 9, 7]
    header = ["case", "runs", "checks", "perfect", "critical", "secs"]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for name, rs in by_case.items():
        all_checks = [c for r in rs for c in r["checks"]]
        cells = [
            name,
            len(rs),
            f"{sum(c['ok'] for c in all_checks) / len(all_checks):.2f}",
            f"{sum(all(c['ok'] for c in r['checks']) for r in rs)}/{len(rs)}",
            sum(c["critical"] and not c["ok"] for c in all_checks),
            f"{sum(r['seconds'] for r in rs) / len(rs):.0f}",
        ]
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, widths)))

    print("\nFailures (critical marked !):")
    for name, rs in by_case.items():
        fails = defaultdict(int)
        critical = {}
        for r in rs:
            for c in r["checks"]:
                if not c["ok"]:
                    fails[c["name"]] += 1
                    critical[c["name"]] = c["critical"]
        for check, n in sorted(fails.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<22} {n}/{len(rs)} {'!' if critical[check] else ' '} {check}")

    all_checks = [c for r in runs for c in r["checks"]]
    print(
        f"\nSUMMARY ({len(runs)} runs, {len(by_case)} cases): "
        f"checks={sum(c['ok'] for c in all_checks) / len(all_checks):.3f} "
        f"perfect_runs={sum(all(c['ok'] for c in r['checks']) for r in runs)}/{len(runs)} "
        f"critical_failures={sum(c['critical'] and not c['ok'] for c in all_checks)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3, help="runs per case (default 3)")
    ap.add_argument("--case", action="append", choices=list(CASES), help="only this case (repeatable)")
    ap.add_argument("--allow-shell", action="store_true",
                    help="include code-helper cases, which auto-approve the model's shell commands")
    ap.add_argument("--timeout", type=float, default=900, help="seconds per run (default 900)")
    args = ap.parse_args()

    names = args.case or list(CASES)
    skipped = [n for n in names if CASES[n].get("shell") and not args.allow_shell]
    if skipped:
        print(f"Skipping {', '.join(skipped)}: pass --allow-shell to auto-approve the model's shell commands.")
    names = [n for n in names if n not in skipped]
    if not names:
        sys.exit("error: nothing to run")

    model = os.environ.get("AMADEUS_MAIN_MODEL", "qwen3:8b")
    preflight(model)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS / stamp
    runs = []
    for name in names:
        for i in range(1, args.runs + 1):
            run_dir = out / f"{name}-{i}"
            run_dir.mkdir(parents=True)
            r = run_once(name, CASES[name], run_dir, args.timeout)
            passed = sum(c["ok"] for c in r["checks"])
            print(f"{name} run {i}: {r['status']}, {passed}/{len(r['checks'])} checks, {r['seconds']:.0f}s", flush=True)
            runs.append(r)
            (out / "summary.json").write_text(json.dumps({"model": model, "runs": runs}, indent=2))
    report(runs)
    print(f"\nModel: {model}. Outputs and run logs: {out}")


if __name__ == "__main__":
    main()
