# AMADEUS

Fully local AI coding assistant + codebase RAG. Everything runs through Ollama
on this machine (localhost:11434, RTX 5070). No API keys, no cloud.

## Layout

```
amadeus/amadeus.sh       entry point: TUI by default, one-shot CLI subcommands
amadeus/rag/rag.py       chunk -> embed (jina-embeddings-v2-base-code) -> sqlite-vec; query + rerank
amadeus/rag/eval/        run_eval.py (recall@k, MRR) + fixtures.jsonl (query -> expected files)
amadeus/tui/             Rust/ratatui front end (binary: amadeus-tui)
amadeus/agent/agent.py   autonomous write/run loop behind `amadeus codex`
amadeus/venv/            Python venv (gitignored); deps in rag/requirements.txt
docs/                    GitHub Pages site (index.html)
```

## Workspace (this machine)

- Working copy: `~/gt/amadeus/crew/salmaan`, a Gas Town crew clone. Code lives under `amadeus/`;
  run `make` from the repo root.
- `~/.local/bin/amadeus` symlinks to `amadeus/amadeus.sh`, and `AMADEUS_HOME` (in `~/.bashrc`) points
  at `~/gt/amadeus/crew/salmaan/amadeus`. Launcher and TUI must resolve to the same checkout.
- Gas Town files (`.beads/`, `.claude/commands/{review,done,handoff}.md`) are excluded via
  `.git/info/exclude`, not `.gitignore`, so they never reach GitHub.
- Open problems are tracked as beads: rig work in `bd -C ~/gt/amadeus list`, town-level in
  `bd -C ~/gt list`. The list at the bottom of this file mirrors them.

## Index data dir (`AMADEUS_RAG_DATA_DIR`)

- The standard index is `~/.amadeus-notest` (test files excluded). Every baseline is measured
  against it. It's set in `~/.bashrc`, and `make` falls back to it.
- Unset, `rag.py` uses `./.amadeus` relative to the cwd, so an index lands wherever you're standing.
  Keep the variable set.
- Experiments that change chunks build into a scratch dir: `make index DATA=~/.amadeus-<topic>`,
  then `make eval DATA=~/.amadeus-<topic>`. Don't rebuild `~/.amadeus-notest` to try something.

## Commands

```
make setup    venv + pip install + cargo fetch
make eval     retrieval eval against $AMADEUS_RAG_DATA_DIR (default ~/.amadeus-notest)
make index    rebuild that index from ~/rag-corpus/gastown/internal
make tui      cargo run --release
make test     cargo test + Python byte-compile
make lint     cargo fmt --check + clippy -D warnings
make check    lint + test + eval
```

Override the index with `make eval DATA=/path/to/index-dir`.

## Eval baseline

**Re-entry check:** `make eval` (runs `rag/eval/run_eval.py` with the venv and
`AMADEUS_RAG_DATA_DIR`) must reproduce this table before you change anything. It needs Ollama to
serve `unclemusclez/jina-embeddings-v2-base-code`; an HTTP 404 from Ollama means that model is
missing or unreadable.

Gas Town Go corpus (`~/rag-corpus/gastown/internal`, tests skipped, 11,094 chunks), 8 fixtures:

| recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| 0.500 | 0.812 | 0.938 | 0.938 | 0.812 |

Reproduced 2026-10-01 from a fresh index build.

## Rules

- **Measure every retrieval change.** Run `make eval` before and after any change to
  chunking, embedding, ranking, filtering or the index. Put both numbers in the commit message.
- **Record negative results.** A change that doesn't help still gets written up
  (README "Retrieval evaluation"), with the numbers. Don't quietly revert it.
- **No cloud dependencies.** No hosted APIs, no hosted vector DBs, no telemetry. Ollama + SQLite only.
- **Honest framing.** README and docs claim only what the eval shows. Small fixture set = say so.
- **Conventional commits:** `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `perf:`; optional scope, e.g. `feat(rag):`.
- Don't edit the production index (`~/.amadeus-notest`) by hand; rebuild it with `make index`.
- **Verify edits landed.** nvim saves sometimes fail silently. Before running anything against an
  edited file, confirm its contents with `git diff <file>` or `sed -n '<a>,<b>p' <file>`.

## Town revival (after a reboot or a long break)

1. **Dolt up.** `gt dolt status`. If stopped, read `gt dolt logs -n 50` for a crash, then
   `gt dolt start`. If it's hung, follow the diagnostics-first steps in `~/gt/CLAUDE.md`; don't
   restart blind.
2. **bd reads.** `bd -C ~/gt list --flat`. "table not found" or "migrations pending" means bd was
   upgraded: note each database's `dolt_log` HEAD, then run `bd -C ~/gt migrate schema` and
   `bd -C ~/gt/amadeus migrate schema`.
3. **Status.** `gt status`, `gt agents check`, `gt dog health-check`. Reset a zombie dog with
   `gt dog clear <name>`.
4. **Workspace clean.** In `~/gt/amadeus/crew/salmaan`: `git status -sb`, `git pull --ff-only`. If
   `.beads/redirect` is gone, recreate it: `mkdir -p .beads && echo ../../.beads > .beads/redirect`.
5. **Eval.** `make eval` matches the baseline above.

## Open problems

1. **Indexing aborts on oversized chunks, and loses everything** (`amadeus-mxs`). A single-line file
   (`docs/design/convoy/stage-launch/bv-insights.json`, 82 KB) becomes one 44k-token chunk. Ollama ≥0.32
   returns HTTP 500 instead of truncating, and `cmd_index` only commits at the end, so the whole
   run is lost (this is what broke `~/.amadeus-notest` on 2026-09-15). Fix: hard-cap chunk length in
   `split_fixed`, and commit per file. Measure, since it changes chunks.
2. **Fixtures: 8 → 20+, plus 3–5 negative queries** (`amadeus-c12`). Eight queries is too few to
   trust small deltas.
3. **`DEF_RE` misses Go `type`/`struct`/`interface` declarations** (`amadeus-th1`), so those land
   inside neighbouring chunks.
4. **Multi-concept dilution** (`hq-lgi`). "how are convoys stored in dolt" retrieves convoy code but
   not the storage layer; the embedding averages concepts. Already failed: lexical rerank, IDF
   weighting. Candidates: query decomposition, model reranker.
5. **Standalone CLI** (`amadeus-wyd`). `amadeus.sh` resolves its home from the script path, the TUI
   from `AMADEUS_HOME`; pick one rule for both.
6. **README is missing the TUI** (`amadeus-7ba`), still says `rag/venv` (the code uses
   `amadeus/venv`), and doesn't mention `qwen2.5-coder:1.5b` isn't pulled by default here.
7. `run_eval.py` `retrieve_ranked_docs` has a duplicated nested loop and returns `None` when no
   rows come back (`amadeus-71i`).
