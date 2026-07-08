#!/usr/bin/env python3
"""
agent.py — AMADEUS execution agent.

A small autonomous loop: the model plans, writes files, runs commands,
reads the output, and iterates until done. Called from amadeus.sh as:

    python3 agent.py "<task>" --model qwen2.5-coder:3b

The model must respond with a single JSON action per turn:
  {"action": "write_file", "path": "...", "content": "..."}
  {"action": "run",        "command": "..."}
  {"action": "done",       "summary": "..."}

Safety: file writes are confined to the launch directory, and every
command is shown and confirmed before execution unless --yes is passed.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("AMADEUS_OLLAMA_URL", "http://localhost:11434")
WORKSPACE = Path.cwd().resolve()

SYSTEM_PROMPT = """You are AMADEUS-EXEC, a software engineering agent that \
completes tasks by writing files and running shell commands.

You operate in a loop. On each turn, respond with EXACTLY ONE JSON object \
and nothing else — no markdown, no backticks, no commentary:

{"action": "write_file", "path": "relative/path.py", "content": "full file content"}
{"action": "run", "command": "shell command"}
{"action": "done", "summary": "what you accomplished"}

Rules:
- Write complete files, never placeholders or diffs.
- After writing code, run it or its tests to verify before declaring done.
- If a command fails, read the error in the next observation and fix it.
- Use relative paths only. Stay inside the current directory.
- Prefer small verifiable steps over one giant leap."""


def chat(model: str, messages: list[dict]) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["message"]["content"]
    except urllib.error.URLError as e:
        sys.exit(f"error: cannot reach Ollama at {OLLAMA_URL} ({e})")


def parse_action(raw: str) -> dict | None:
    """Extract the first JSON object from the model's reply. Small models
    sometimes wrap JSON in backticks despite instructions — tolerate it."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    # Find the first balanced {...} block.
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def safe_path(rel: str) -> Path:
    """Resolve a model-supplied path and refuse anything that escapes
    the workspace (absolute paths, ../ tricks, symlink hops)."""
    target = (WORKSPACE / rel).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"path escapes workspace: {rel}")
    return target


def do_write(action: dict) -> str:
    try:
        target = safe_path(action["path"])
    except ValueError as e:
        return f"REFUSED: {e}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(action["content"])
    lines = action["content"].count("\n") + 1
    print(f"  wrote {action['path']} ({lines} lines)")
    return f"OK: wrote {action['path']} ({lines} lines)"


def do_run(action: dict, auto_yes: bool) -> str:
    cmd = action["command"]
    print(f"  $ {cmd}")
    if not auto_yes:
        answer = input("  run this? [y/N] ").strip().lower()
        if answer != "y":
            return "USER DECLINED: command was not run."
    result = subprocess.run(
        cmd, shell=True, cwd=WORKSPACE,
        capture_output=True, text=True, timeout=300,
    )
    out = (result.stdout + result.stderr).strip()
    out = out[-3000:]  # keep observations small for small models
    print(out or "  (no output)")
    return f"exit code {result.returncode}\n{out}"


def main() -> None:
    parser = argparse.ArgumentParser(description="AMADEUS execution agent")
    parser.add_argument("task")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--yes", action="store_true",
                        help="run commands without confirmation")
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {args.task}\nWorking directory: {WORKSPACE}"},
    ]

    for step in range(1, args.max_steps + 1):
        print(f"\n[step {step}/{args.max_steps}]")
        reply = chat(args.model, messages)
        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        if action is None:
            observation = ("Your reply was not a single valid JSON action. "
                           "Respond with exactly one JSON object.")
            print("  (unparseable reply, asking model to retry)")
        elif action.get("action") == "done":
            print(f"\ndone: {action.get('summary', '(no summary)')}")
            return
        elif action.get("action") == "write_file":
            observation = do_write(action)
        elif action.get("action") == "run":
            observation = do_run(action, args.yes)
        else:
            observation = f"Unknown action: {action.get('action')!r}"

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    print("\nstopped: reached max steps without a done action")


if __name__ == "__main__":
    main()