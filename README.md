# AMADEUS

A fully local AI task harness with codebase-grounded retrieval. No API keys, no subscriptions, no cloud. Every model runs on my own hardware through [Ollama](https://ollama.com), and nothing leaves the machine.

AMADEUS is a terminal app. You pick a task template or type a task, and a local model works through it with a small set of tools. It can read files on its own, but every file write and shell command stops at an approval card showing the exact diff or command. Each run is logged so you can see what happened afterwards.

```
amadeus                open the TUI in the current directory
amadeus doctor         check Ollama, models, GPU, venv and build; prints the fix for each problem
amadeus ask <task>     one-shot answer, streamed to stdout
amadeus rag <question> one-shot answer grounded in the RAG index
amadeus index [dir]    build or refresh the RAG index
amadeus agent <task>   TUI on the code-helper template, running <task>
amadeus replay [log]   readable transcript of a task's run log
```

## Quick start

You need [Ollama](https://ollama.com/download), a Rust toolchain and Python 3. It's been run on one machine so far: WSL2 on an RTX 5070 Laptop GPU (8 GB).

```bash
make setup     # venv, Python deps, Rust deps
make models    # qwen3:8b + qwen3:4b-instruct, about 8 GB
make demo      # build, run the doctor, open the TUI on a fresh copy of demo/workspace
```

`make demo` works on a temp copy, so the repo never collects demo output. [`demo/DEMO.md`](demo/DEMO.md) is the walkthrough.

`make tui` builds and runs the TUI on its own. The launcher also rebuilds it whenever the sources are newer than the binary.

To run it from anywhere, link the launcher onto your `PATH`:

```bash
ln -s "$PWD/amadeus/amadeus.sh" ~/.local/bin/amadeus
```

Code search needs the embedding model and an index:

```bash
ollama pull unclemusclez/jina-embeddings-v2-base-code
amadeus index path/to/repo
```

## Templates

A template is a TOML file with a system prompt, a tool list and a step limit. Four ship in `amadeus/templates/`. A file in `~/.config/amadeus/templates/` with the same `name` replaces the shipped one.

| Template | What it does | Tools |
|---|---|---|
| `folder-summary` | Reads the documents in a folder and writes a briefing | list, read, write |
| `notes-to-spreadsheet` | Turns messy notes into a CSV | list, read, write |
| `draft-replies` | Drafts a reply to each message and saves it as a file. Nothing is sent | list, read, write |
| `code-helper` | Answers questions about a codebase and changes it | list, read, search, RAG, write, run |

### How well they work

I ran all four once on `qwen3:8b` on 2026-11-04 with the demo files. One run per template is a smoke test, not a benchmark. Each one finished in 8 to 18 seconds with the model already loaded, and the harness did its job every time. The answers were a different matter:

| Template | Result | What was wrong |
|---|---|---|
| `code-helper` | Correct | Nothing. It fixed a planted bug and ran the check, which exited 0 |
| `folder-summary` | Usable, with errors | Listed gift wrapping as an open question after a later note had decided it. Only 3 of the 9 action items made it in. Decisions had no source files |
| `notes-to-spreadsheet` | Numbers and dates right, vendors not | Made up four vendors that the notes don't name, such as "Team Sync", despite the instruction to leave them empty |
| `draft-replies` | Not sendable | One reply refused a refund the policy allows for a wrong item. One promised delivery in time for a birthday, which the policy forbids. One asked for details the customer had already given |

The task eval in `amadeus/eval/` measures this with scored runs on fictional HR files. Its 2026-11-07 baseline on `qwen3:8b` passed 79% of checks, and 0 of 12 runs were perfect. In every run of the reply case, the model put a coworker's pay in a draft and followed a prompt injection planted in an email. Details are in [`amadeus/eval/README.md`](amadeus/eval/README.md).

So treat the output as a first draft that needs checking, which is why the approval card exists. A decline can carry a reason, and the model does act on it. In the spreadsheet run, "leave paid_by empty when a line does not name who paid" fixed the one cell it applied to. It only fixes what you notice, though: the invented vendors stayed because I didn't mention them.

## How it works

```
amadeus.sh        resolves AMADEUS_HOME from its own path, rebuilds the TUI if stale
   │
   └─ amadeus-tui (Rust, single process)
        ├── agent      the one loop: /api/chat stream → tool calls → approval → result → repeat
        ├── tools      read_file, list_dir, search_files, write_file, run_command, rag_query
        ├── approval   y approve · n skip and say why · a approve the rest of this task
        ├── templates  TOML task definitions
        ├── runlog     one JSONL file per task
        └── doctor     machine readiness checks
   │
   └─ rag.py (Python): spawned for rag_query and /index, killed on cancel
```

What holds it together:

- **Approvals.** Reads run straight away. `write_file` and `run_command` wait for you. If you decline, your reason goes back to the model, and an exact repeat of a declined call is refused without asking again. `/auto` turns approvals off for the next task only.
- **Workspace confinement.** The workspace is the directory you launched in. File tools refuse absolute paths, `..` and symlinks that lead outside it. `run_command` is a shell and isn't confined, so it always needs approval unless you've used `a` or `/auto`.
- **Real cancel.** Ctrl+C closes the connection to Ollama, so generation stops on the server and doesn't keep holding the GPU. The custom HTTP client this needs speaks plain `http://` only, so `AMADEUS_OLLAMA_URL` can't be an `https://` address.
- **Run logs.** Every turn, tool call, approval and result goes to `~/.local/state/amadeus/projects/<slug>/runs/`. `/log` shows the path and `amadeus replay` prints it readably.
- **One context size.** Every request sends the same `num_ctx`, because Ollama reloads a model when it changes. `num_predict` caps a single turn, and the TUI tells you if a reply was cut off.

The design record, with the protocol measurements behind these choices, is in [`docs/REVAMP.md`](docs/REVAMP.md).

### Keys and commands

`?` shows everything. The ones you'll use most:

| Key or command | Does |
|---|---|
| `/new`, `/use <name>` | pick a template |
| `/chat`, `/rag`, `/ask` | plain chat, indexed code, one question to the model |
| `/auto` | auto-approve the next task |
| `/file <path>` | stage a file into the next chat |
| `/log` | where the last task's audit log is |
| `ctrl+c` | stop the current task |
| `ctrl+t` | cycle the model: fast / main / heavy |
| `ctrl+o` | show or hide the model's reasoning |

## Why local, and why tiers

This project started under a real constraint: a convertible laptop with no dedicated GPU and 8GB of RAM. That ruled out large local models, and I didn't want to depend on paid cloud APIs for everyday coding help. The constraint shaped the design: send each task to the smallest model that can handle it.

| Tier | Default model | Size | Used for |
|---|---|---|---|
| fast | qwen3:4b-instruct | 2.5 GB | Quick questions, weaker hardware |
| main | qwen3:8b | 5.2 GB | Every shipped template, RAG answers |
| heavy | qwen3:14b | 9.3 GB | 12 GB cards and up. On 8 GB it spills to CPU. Not pulled by `make models` |

It has since moved to a machine with a GPU (Intel Core Ultra 9 275HX, 32GB RAM, RTX 5070), which changed what's practical — indexing 20,000 chunks takes about 16 minutes with GPU-backed embeddings instead of hours on CPU.

Tiers are no longer routed by typed commands. A template names its tier, and `ctrl+t` or `--fast` / `--heavy` switch it. Two measurements set the defaults:

- **`rag` uses the main model.** On the same grounded prompt, `qwen2.5-coder:3b` ignored the citation instruction entirely and invented a mechanism that wasn't in the retrieved context. `qwen3:8b` cited every claim and stayed inside the context.
- **The agent loop needs native tool calls.** `qwen2.5-coder` writes its calls as JSON text in the reply, while `qwen3:8b` returns real `tool_calls`. The harness parses both, but the templates are written for the second.

## RAG: codebase-grounded answers

`amadeus/rag/rag.py` indexes a repository and answers questions grounded only in that code — a fully local pipeline with zero API dependencies.

Chunks are cut on definition boundaries (`def`, `class`, `func`, `impl`…) rather than fixed character windows, because slicing a function mid-token degrades the embedding. Anything larger than the chunk budget falls back to line-aligned windows with overlap. Each chunk is embedded with `jina-embeddings-v2-base-code` through Ollama, and vectors live in a single SQLite file via [sqlite-vec](https://github.com/asg017/sqlite-vec) — no vector database server eating RAM.

Design decisions worth noting:

- **A code-trained embedding model matters more than the chunker.** The pipeline originally used `nomic-embed-text`. Fixing chunk boundaries changed retrieval rankings by exactly nothing — same eight chunks, same distances to four decimals. Swapping to a code-trained model was the change that moved rankings. Prose embeddings can't separate an argparse block from a KNN query.
- **Relevance is filtered by a relative margin, not an absolute threshold.** Distance bands shift per query — 0.31–0.37 on one question, 0.42–0.48 on another — so a fixed cutoff admits everything on some queries and nothing on others. `AMADEUS_RAG_MARGIN` keeps hits within a margin of the closest match. This controls how *wide* the context is; it can't reject an off-topic question, and the honest position is that relevance isn't filterable at the distance layer with this model.
- **Test files are excluded by default.** In one Go corpus, `_test.go` files were 44% of all chunks and crowded implementation out of the results entirely — a question about heartbeat staleness returned eight test files and zero implementation. `AMADEUS_RAG_SKIP_TESTS=0` restores them.
- **Location is baked into the vectors.** Every chunk is embedded with a `# File: path, lines N-M` header, so questions that mention a module name match its chunks even when the code never uses that word.
- **Incremental re-indexing.** File hashes in `index_meta.json` mean unchanged files are skipped and deleted files are purged. Deleting `index.db` invalidates the hashes too — otherwise the index reports "N files unchanged" and silently rebuilds nothing.
- **The index location is explicit.** `AMADEUS_RAG_DATA_DIR` sets where the index lives, so several corpora can coexist and queries aren't silently answered by whichever index happens to sit in the current directory.

## Retrieval evaluation

`amadeus/rag/eval/run_eval.py` scores retrieval against `amadeus/rag/eval/fixtures.jsonl` — question → expected-file pairs — and reports recall@k and MRR at the file level. It calls the same `embed()` and rerank path the CLI uses, so it measures what actually ships.

Current baseline, 25 queries over the Gas Town Go codebase (1,231 files, 11,104 chunks after test exclusion):

| recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| 0.380 | 0.640 | 0.720 | 0.800 | 0.592 |

Twenty-five queries is still a small set; treat differences of a few hundredths as noise. Four negative queries (no answer in the corpus) sit in `amadeus/rag/eval/fixtures_negative.jsonl` and aren't scored yet.

What the harness has settled so far:

- **Excluding test files** moved MRR from 0.688 to 0.812 and recall@5 from 0.750 to 0.938. The heartbeat query went from 0.00 at every k to 1.00 at k=5.
- **Lexical reranking doesn't help.** Blending term-overlap into the ranking degraded results at every weight from 0.02 to 0.30 (recall@1 0.500 → 0.438, MRR 0.812 → 0.729).
- **Neither does IDF weighting.** The hypothesis was that common tokens were drowning rare ones. Rarity weighting was swept from 0.005 to 0.05 and degraded results just as consistently. Term presence at 1500-character chunk granularity carries little relevance signal regardless of how it's weighted. `AMADEUS_RAG_LEXICAL_WEIGHT` defaults to `0.0`; the `rerank()` hook stays as the measured entry point for future attempts.
- **Capping chunk length is retrieval-neutral.** Lines longer than 1,500 characters are now hard-split, so no chunk exceeds the cap (the largest chunk went from 6,031 to 1,500 characters; 11,094 → 11,104 chunks). Every metric and every per-query score is unchanged. The cap is a robustness fix: Ollama ≥0.32 rejects oversized inputs with HTTP 500 instead of truncating, and one 82 KB single-line JSON file used to abort a whole indexing run. Indexing now also commits per file, so a failure loses only the file in progress.
- **The first eight fixtures were optimistic.** Growing the set from 8 to 25 queries, with the index unchanged, dropped MRR from 0.812 to 0.592 and recall@5 from 0.938 to 0.720. Three of the new queries miss entirely at k=10: MR priority scoring (`refinery/score.go`), the identity lock (`lock/lock.go`) and the nudge poller for non-Claude runtimes (`nudge/poller.go`). In each case retrieval returns neighbouring files that use the concept rather than the one that implements it. The earlier numbers in this list were measured on the 8-query set.
- **Splitting chunks at Go `type` declarations hurts.** Adding `type ` to the definition regex (`amadeus-th1`) took the index from 11,104 to 12,060 chunks and dropped MRR from 0.592 to 0.493 (recall@1 0.380 → 0.280, recall@5 0.720 → 0.660, recall@10 0.800 → 0.720). Eight queries lost rank and one gained; none of the three zero-hit queries moved into the top 10, and two of their target files ranked lower. The cause is where the boundary falls: at the `type` line itself, so the doc comment above it stays in the previous chunk and the struct chunk loses its description. "How does the daemon limit restarts of crashing agents" went from rank 1 to outside the top 10 because `restart_tracker.go`'s winning chunk had been the comment and struct together. Reverted. `func` boundaries split comments off the same way, so the next attempt moves every boundary up to the start of its leading comment block.
- **Multi-concept queries collapse to the dominant term.** "How are convoys stored in dolt" returns convoy display code and no storage layer, while "dolt sql server connection" alone returns the storage layer cleanly. Both concepts are indexed; the embedding averages rather than conjoins. Still open — candidates are query decomposition or a model-based reranker.

The negative results are the point. Four of the five changes tried didn't help, and measuring took minutes instead of an afternoon of arguing with the output.

## Repo layout

```
amadeus/
├── amadeus.sh          # launcher: TUI by default, one-shot subcommands
├── tui/                # Rust/ratatui front end and agent loop (binary: amadeus-tui)
├── templates/          # task templates (TOML)
├── rag/
│   ├── rag.py          # chunk, embed, store, retrieve, rerank
│   ├── eval/
│   │   ├── run_eval.py     # recall@k + MRR against fixtures
│   │   └── fixtures.jsonl  # question → expected-file pairs
│   └── requirements.txt
├── eval/               # task eval: scored template runs on fictional HR data
└── venv/               # Python venv (gitignored)
demo/
├── DEMO.md             # walkthrough
└── workspace/          # fictional sample files for make demo
docs/
├── index.html          # project site
└── REVAMP.md           # design record for the harness
```

`make help` lists the targets. `make check` runs lint, tests and the retrieval eval.

## Configuration

Everything is overridable with environment variables:

```bash
AMADEUS_FAST_MODEL=qwen3:4b-instruct \
AMADEUS_MAIN_MODEL=qwen3:8b \
AMADEUS_HEAVY_MODEL=qwen3:14b \
AMADEUS_OLLAMA_URL=http://localhost:11434 \
AMADEUS_NUM_CTX=8192 \
AMADEUS_NUM_PREDICT=3072 \
AMADEUS_MOUSE=1 \
AMADEUS_RAG_DATA_DIR=.amadeus \
AMADEUS_RAG_TOP_K=8 \
AMADEUS_RAG_MARGIN=0.06 \
AMADEUS_RAG_SKIP_TESTS=1 \
AMADEUS_RAG_LEXICAL_WEIGHT=0.0 \
amadeus
```

`AMADEUS_MOUSE=0` gives you native text selection back. `amadeus/rag/rag.py query "your question" --debug` prints raw cosine distances to stderr, which is how the margin gets calibrated for a given codebase.

## What I learned

- Small local models (1.5B–3B) are genuinely useful for coding when tasks are scoped tightly — but not for grounded citation, where they silently ignore the instruction and fill gaps from training data.
- Prompt assembly, history windowing, and model selection are the real plumbing behind tools like Copilot. Building it by hand made that visible.
- Retrieval intuitions are usually wrong. Chunk boundaries looked like the obvious problem and weren't; test files looked like noise and were 44% of the corpus. An eval harness with a dozen verified question→file pairs settles in seconds what eyeballing output can't settle at all.
- A harness that reimplements the retrieval path instead of calling it will happily report good numbers for code that's broken in production. Ask me how I know.

## Roadmap

- [x] Local RAG over the codebase (sqlite-vec + code-trained embeddings)
- [x] Retrieval evaluation harness: question→expected-file pairs, recall@k and MRR
- [x] Grow the fixture set past 20 pairs so smaller effects are detectable
- [x] One Rust agent loop with approvals, confinement, templates and run logs
- [x] Streaming responses and cancel that stops generation on the server
- [ ] Make template output more accurate, and measure it the way retrieval is measured
- [ ] Fix multi-concept query dilution (decomposition or model-based reranking)
- [ ] Score the negative queries, so the CLI distance threshold is tested
- [ ] Semantic session memory: embed history and retrieve relevant past exchanges instead of the blind last-40-lines window
