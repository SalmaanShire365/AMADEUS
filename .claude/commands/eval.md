---
description: Run the retrieval eval and compare against the CLAUDE.md baseline
allowed-tools: Bash(make eval), Bash(make eval *), Bash(git diff *), Bash(git status *)
argument-hint: [DATA=/path/to/index-dir]
---

Run `make eval $ARGUMENTS` from the repo root.

Then report:

1. The SUMMARY line.
2. A table comparing recall@1, recall@3, recall@5, recall@10 and MRR against the baseline in
   CLAUDE.md, with the delta for each metric.
3. Per-query rows whose RR changed, if a previous run is visible in this conversation.
4. If `git diff` touches `amadeus/rag/`, a one-line verdict: better / worse / no change,
   ready to paste into the commit message. A worse result is still recorded (see Rules).

If the eval refuses to run (missing index, leftover journal), report the message and stop.
Don't rebuild the index without asking.
