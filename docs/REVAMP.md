# AMADEUS revamp: local task harness

Branch `revamp/harness`. This is the plan written before the work started, followed by the decisions as
they landed. It stays in the repo as the design record.

## Baseline (2026-10-25, before any change)

- `make test`: 16 Rust tests pass, Python byte-compiles.
- `make lint`: `cargo fmt --check` and `clippy -D warnings` clean.
- `make eval` against `~/.amadeus-notest`: recall@1 0.380, @3 0.640, @5 0.720, @10 0.800, MRR 0.592.
  Matches CLAUDE.md exactly.
- Machine: Ollama 0.40.10, RTX 5070 **Laptop, 8 GB**. That's below the 12–16 GB demo target, so the
  defaults are chosen to fit 8 GB and a bigger card only makes them faster.

## What I measured before choosing a protocol

One `/api/chat` call with three tools (`list_dir`, `read_file`, `write_file`) and the prompt "Summarize
the meeting notes in the notes/ folder":

| model | native `tool_calls` | what it did instead |
|---|---|---|
| qwen2.5-coder:3b | no | wrote two `{"name":…, "arguments":…}` objects into `content` |
| qwen2.5-coder:7b | no | one JSON call object in `content` |
| qwen3:8b, `think:false` | **yes** | `list_dir {"path":"notes"}`, empty content, no reasoning |
| qwen3:8b, default | yes | same call, plus a `thinking` field |

`/api/show` reports `tools` for all four, so the capability flag can't be trusted to mean "emits
structured calls". `think:false` sent to a non-thinking model is accepted silently.

Consequences:

- **Native tool calling on `/api/chat`** is the primary path. When a reply has no `tool_calls` but its
  content is one or more JSON objects of the form `{"name", "arguments"}`, the harness parses them as
  calls. That keeps the qwen2.5-coder models usable for `code-helper` without a second protocol.
- **`think:false`** is sent whenever `/api/show` lists `thinking`. Any `<think>…</think>` that still
  shows up in content (older models, other runtimes) is folded into one dimmed line in the scrollback.
- **Default tiers:** fast `qwen3:4b-instruct` (2.5 GB), main `qwen3:8b` (5.2 GB), heavy `qwen3:14b`
  (9.3 GB, for 12–16 GB cards; on 8 GB it spills to CPU). All three are overridable with
  `AMADEUS_{FAST,MAIN,HEAVY}_MODEL`.
- **Not `qwen3:4b`.** That tag is now the Qwen3-4B *Thinking*-2507 build. Its chat template opens
  `<think>` inside the prompt, so `think:false` can't turn it off, and the reasoning streams into
  `content` with only a closing `</think>`. Measured: the probe above returned two paragraphs of
  reasoning in `content` alongside a correct tool call. `qwen3:4b-instruct` (Instruct-2507) gives the
  same tool call with empty content in 8 s.

## Architecture

```
amadeus.sh ── resolves AMADEUS_HOME from its own path, exports it, rebuilds the TUI if stale
   │
   └─ amadeus-tui (Rust, single process)
        ├── ui         header bar, scrollback (markdown, folded <think>), approval card, help overlay
        ├── app        state + event routing; input editor with Up/Down history
        ├── scrollback entries + wrap cache that tolerates a live entry that isn't last
        ├── agent      the one agent loop: /api/chat stream → tool calls → approval → result → repeat
        ├── tools      registry: read_file, list_dir, search_files, write_file, run_command, rag_query
        │              every path goes through workspace confinement; each tool declares needs_approval
        ├── templates  TOML files in $AMADEUS_HOME/templates (and ~/.config/amadeus/templates)
        ├── runlog     JSONL per task under $XDG_STATE_HOME/amadeus/projects/<slug>/runs/
        ├── proc       killable child processes (rag.py, run_command, /shell) with output streaming
        └── doctor     Ollama reachable, models pulled, GPU via /api/ps, venv, TUI build
   │
   └─ rag.py (Python, unchanged): spawned as a child for rag_query and /index, killed on cancel
```

### Decision: the agent loop lives in Rust, `agent.py` is removed

- **Approval is a UI interaction.** The loop has to stop, show a diff or a command, and wait for y / n / a.
  In-process, that's a channel send and a blocking receive. Across a process boundary it's a
  bidirectional JSON-lines protocol with a pending-request state machine on both sides, for no gain.
- **Cancel is one flag.** Ctrl+C sets the run's cancel flag, and the loop checks it while it waits on the
  model, the approval, or a child process. There's no second process to signal and reap.
- **The templates a buyer sees don't need Python.** folder-summary, notes-to-spreadsheet and draft-replies
  run on the Rust binary and Ollama alone. Python is only needed for `rag_query` (code-helper).
- **It's testable with `cargo test`.** The approval state machine, the tool confinement and the call
  parser are pure functions, tested without a terminal or a model.

`amadeus agent <task>` (and the old alias `codex`) now opens the TUI with the code-helper template and
the task pre-submitted, so there's one loop and one approval UI.

### Approval model

Every tool declares `needs_approval`. Reads (`read_file`, `list_dir`, `search_files`, `rag_query`) run
straight away. Writes and commands (`write_file`, `run_command`) stop the loop and show a card:
the command, or a unified diff of the file against what's on disk. The keys are **y** (run it), **n**
(skip it, and the model is told it was declined), and **a** (approve this and the rest of this run).
`/auto` turns auto-approve on for the next run only. Nothing is hard-coded to `--yes`.

### Workspace confinement

The workspace is the directory the TUI was launched in. Every tool path is resolved against it and
refused if it escapes (absolute paths, `..`, symlinks pointing outside), the same rule as
`agent.py`'s `safe_path`. For a path that doesn't exist yet, the nearest existing ancestor is resolved.
`run_command` runs with the workspace as cwd, stdin closed, a timeout, and capped output. It's a
shell, so it isn't confined, and that's why it always needs approval.

### State dir

`$XDG_STATE_HOME/amadeus/` (fallback `~/.local/state/amadeus/`):

```
input_history          Up/Down recall, global like a shell's
projects/<slug>/
  history.txt          chat context, was <cwd>/.agent_history.txt
  runs/<ts>-<template>.jsonl   one file per task: every turn, tool call, approval, result
```

`<slug>` is the workspace path with separators flattened, plus a 64-bit FNV-1a hash of the full path, so
`~/a/b` and `~/a_b` never collide and the name is still readable.

## Phases

0. Baseline + this plan.
1. TUI fixes: streaming target, real cancel, `<think>`, history location, launcher, basics (history,
   mouse, markdown, health check, source display). The `--yes` fix needs the approval UI, which needs
   the Rust loop, so it lands at the start of Phase 2 instead of as throwaway code in `agent.py`.
2. Harness core: tools, templates, the Rust loop with approvals, run log. Delete `agent.py`.
3. Demo: `amadeus doctor`, `demo/` workspace, `make demo`, DEMO.md, VHS tape, header bar, help overlay.
4. Docs + an end-to-end run of DEMO.md on the real models.

## Dependencies considered

| project | license | verdict |
|---|---|---|
| tui-textarea 0.7.0 | MIT | not used: no soft wrap (long lines scroll sideways, a regression from the current hard-wrapping input box), and Up/Down history still has to be written either way. The existing editor got history and line movement instead |
| pulldown-cmark 0.13 | MIT | **used**: markdown parsing for the scrollback renderer |
| toml 0.9 + serde 1 | MIT/Apache-2.0 | **used**: template files |
| tui-markdown 0.3.12 | MIT/Apache-2.0 | not used: needs ratatui 0.30, and our wrap cache needs per-line control anyway |
| throbber-widgets-tui 0.11 | Zlib | not used: needs ratatui 0.30, Zlib is outside MIT/Apache/BSD, and a spinner is 3 lines |
| crossterm mouse capture | MIT | **used** (already a dependency): wheel scroll. `AMADEUS_MOUSE=0` turns it off to get native selection |
| charmbracelet/vhs | MIT | **used** for `demo/amadeus.tape`, an optional external tool, not a build dependency |
| block/goose, huggingface/smolagents | Apache-2.0 | design reference only: goose's per-tool approval modes, smolagents' tool spec (name, description, JSON-schema inputs). No code copied |
