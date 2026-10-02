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

## Open problems

1. **Indexing aborts on oversized chunks, and loses everything.** A single-line file
   (`docs/design/convoy/stage-launch/bv-insights.json`, 82 KB) becomes one 44k-token chunk. Ollama ≥0.32
   returns HTTP 500 instead of truncating, and `cmd_index` only commits at the end, so the whole
   run is lost (this is what broke `~/.amadeus-notest` on 2026-09-15). Fix: hard-cap chunk length in
   `split_fixed`, and commit per file. Measure, since it changes chunks.
2. **Fixtures: 8 → 20+.** Eight queries is too few to trust small deltas.
3. **`DEF_RE` misses Go `type`/`struct`/`interface` declarations**, so those land inside
   neighbouring chunks.
4. **Multi-concept dilution.** "how are convoys stored in dolt" retrieves convoy code but not the
   storage layer; the embedding averages concepts. Candidates: query decomposition, model reranker.
5. **Standalone CLI.** Make `amadeus` runnable from anywhere via `AMADEUS_HOME` instead of the script dir.
6. **README is missing the TUI**, still says `rag/venv` (the code uses `amadeus/venv`), and
   doesn't mention `qwen2.5-coder:1.5b` isn't pulled by default here.
7. `run_eval.py` `retrieve_ranked_docs` has a duplicated nested loop and returns `None` when no
   rows come back.
