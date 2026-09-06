# AMADEUS

A fully local, tiered AI coding assistant with codebase-grounded retrieval. No API keys, no subscriptions, no cloud. Every model runs on my own hardware through [Ollama](https://ollama.com).

AMADEUS is an interactive terminal session that routes each task to a different local model depending on difficulty, keeps rolling session memory, lets you inject files into the prompt, answers questions grounded in an indexed codebase, and can hand off to a separate autonomous Python agent when a task needs real code execution instead of just code generation.

```
=== AMADEUS ===
Project: /home/salma
Session: 1783372494
Commands:
  file <path>        → include file in next task
  fast <task>        → quick model (qwen2.5-coder:1.5b)
  heavy <task>       → strong model (qwen3:8b)
  index [dir]        → build/refresh RAG index
  rag <question>     → codebase-grounded answer
  codex <task>       → real code execution via local agent
  shell <cmd>        → run a shell command
  history            → show session history
  clear              → clear history file
  exit               → quit
```

## Why local, and why tiered routing

This project started under a real constraint: a convertible laptop with no dedicated GPU and 8GB of RAM. That ruled out large local models, and I didn't want to depend on paid cloud APIs for everyday coding help.

The constraint shaped the design. Instead of one big model, AMADEUS uses **tiered routing** — send each task to the smallest model that can handle it.

| Tier | Model | Size | Used for |
|---|---|---|---|
| `fast` | qwen2.5-coder:1.5b | ~1GB | Quick lookups, one-liners, syntax questions |
| default | qwen2.5-coder:3b | ~2GB | Everyday coding tasks |
| `heavy` | qwen3:8b | ~5GB | Harder reasoning, architecture questions, all RAG answers |

It has since moved to a machine with a GPU (Intel Core Ultra 9 275HX, 32GB RAM, RTX 5070), which changed what's practical — indexing 20,000 chunks takes about 16 minutes with GPU-backed embeddings instead of hours on CPU. The routing stayed, because it's still the right default: a 1.5B model answers a syntax question instantly and an 8B model doesn't.

One thing the move made measurable rather than assumed: **`rag` now defaults to the heavy model, not the 3B.** On the same grounded prompt, `qwen2.5-coder:3b` ignored the citation instruction entirely and invented a mechanism that wasn't in the retrieved context. `qwen3:8b` cited every claim and stayed inside the context. Use `rag --fast` to opt down when you want speed over grounding.

## Architecture

```
Task> input
   │
   ├── file <path>   → stack file contents into next prompt
   ├── shell <cmd>   → passthrough to the shell
   ├── index [dir]   → chunk + embed + store the codebase
   ├── rag <q>       → retrieve top-k chunks → grounded prompt → model
   ├── codex <task>  → hand off to autonomous local agent
   │
   └── everything else
          │
          ├── fast  → qwen2.5-coder:1.5b
          ├── heavy → qwen3:8b
          └── default → qwen2.5-coder:3b
                 │
                 ▼
        prompt = system prompt
               + last 40 lines of session history
               + any loaded file context
               + task
                 │
                 ▼
             ollama run
                 │
                 ▼
        response → terminal + appended to .agent_history.txt
```

Key mechanics:

- **Session memory.** Every exchange is appended to `.agent_history.txt`, and the last 40 lines are injected into each prompt, giving small stateless models continuity across tasks.
- **Stackable file context.** `file <path>` can be called multiple times; all loaded files are included in the next prompt, then cleared.
- **Execution handoff.** `codex <task>` calls a separate Python agent that can write files, run commands, and review its own output — the generation/execution split keeps the shell script simple.

## RAG: codebase-grounded answers

`rag/rag.py` indexes a repository and answers questions grounded only in that code — a fully local pipeline with zero API dependencies.

Chunks are cut on definition boundaries (`def`, `class`, `func`, `impl`…) rather than fixed character windows, because slicing a function mid-token degrades the embedding. Anything larger than the chunk budget falls back to line-aligned windows with overlap. Each chunk is embedded with `jina-embeddings-v2-base-code` through Ollama, and vectors live in a single SQLite file via [sqlite-vec](https://github.com/asg017/sqlite-vec) — no vector database server eating RAM.

Design decisions worth noting:

- **A code-trained embedding model matters more than the chunker.** The pipeline originally used `nomic-embed-text`. Fixing chunk boundaries changed retrieval rankings by exactly nothing — same eight chunks, same distances to four decimals. Swapping to a code-trained model was the change that moved rankings. Prose embeddings can't separate an argparse block from a KNN query.
- **Relevance is filtered by a relative margin, not an absolute threshold.** Distance bands shift per query — 0.31–0.37 on one question, 0.42–0.48 on another — so a fixed cutoff admits everything on some queries and nothing on others. `AMADEUS_RAG_MARGIN` keeps hits within a margin of the closest match. This controls how *wide* the context is; it can't reject an off-topic question, and the honest position is that relevance isn't filterable at the distance layer with this model.
- **Test files are excluded by default.** In one Go corpus, `_test.go` files were 44% of all chunks and crowded implementation out of the results entirely — a question about heartbeat staleness returned eight test files and zero implementation. `AMADEUS_RAG_SKIP_TESTS=0` restores them.
- **Location is baked into the vectors.** Every chunk is embedded with a `# File: path, lines N-M` header, so questions that mention a module name match its chunks even when the code never uses that word.
- **Incremental re-indexing.** File hashes in `index_meta.json` mean unchanged files are skipped and deleted files are purged. Deleting `index.db` invalidates the hashes too — otherwise the index reports "N files unchanged" and silently rebuilds nothing.
- **The index location is explicit.** `AMADEUS_RAG_DATA_DIR` sets where the index lives, so several corpora can coexist and queries aren't silently answered by whichever index happens to sit in the current directory.

## Retrieval evaluation

`rag/eval/run_eval.py` scores retrieval against `rag/eval/fixtures.jsonl` — question → expected-file pairs — and reports recall@k and MRR at the file level. It calls the same `embed()` and rerank path the CLI uses, so it measures what actually ships.

Current baseline, over the Gas Town Go codebase (1,231 files, 11,094 chunks after test exclusion):

| recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| 0.500 | 0.812 | 0.938 | 0.938 | 0.812 |

What the harness has settled so far:

- **Excluding test files** moved MRR from 0.688 to 0.812 and recall@5 from 0.750 to 0.938. The heartbeat query went from 0.00 at every k to 1.00 at k=5.
- **Lexical reranking doesn't help.** Blending term-overlap into the ranking degraded results at every weight from 0.02 to 0.30 (recall@1 0.500 → 0.438, MRR 0.812 → 0.729).
- **Neither does IDF weighting.** The hypothesis was that common tokens were drowning rare ones. Rarity weighting was swept from 0.005 to 0.05 and degraded results just as consistently. Term presence at 1500-character chunk granularity carries little relevance signal regardless of how it's weighted. `AMADEUS_RAG_LEXICAL_WEIGHT` defaults to `0.0`; the `rerank()` hook stays as the measured entry point for future attempts.
- **Multi-concept queries collapse to the dominant term.** "How are convoys stored in dolt" returns convoy display code and no storage layer, while "dolt sql server connection" alone returns the storage layer cleanly. Both concepts are indexed; the embedding averages rather than conjoins. Still open — candidates are query decomposition or a model-based reranker.

The negative results are the point. Three of the four changes tried didn't work, and measuring took minutes instead of an afternoon of arguing with the output.

## Execution agent

`agent/agent.py` is a small autonomous loop invoked by the `codex` command: the model responds with one JSON action per turn (`write_file`, `run`, or `done`), the agent executes it, and feeds the observation back until the task completes. File writes are confined to the working directory, and every shell command is shown and confirmed before it runs.

## Setup

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Pull the models (adjust to your hardware):
   ```bash
   ollama pull qwen2.5-coder:1.5b
   ollama pull qwen2.5-coder:3b
   ollama pull qwen3:8b
   ollama pull unclemusclez/jina-embeddings-v2-base-code
   ```
3. Set up the RAG module:
   ```bash
   python3 -m venv rag/venv
   rag/venv/bin/pip install -r rag/requirements.txt
   ```
4. Run:
   ```bash
   chmod +x amadeus.sh
   ./amadeus.sh
   ```

## Repo layout

```
amadeus/
├── amadeus.sh          # interactive session: routing, tiers, history
├── rag/
│   ├── rag.py          # chunk, embed, store, retrieve, rerank
│   ├── eval/
│   │   ├── run_eval.py     # recall@k + MRR against fixtures
│   │   └── fixtures.jsonl  # question → expected-file pairs
│   └── requirements.txt
├── agent/
│   └── agent.py        # autonomous plan → write → run → review loop
└── .amadeus/           # per-project index (created at runtime, gitignored)
```

## Configuration

Everything is overridable with environment variables:

```bash
AMADEUS_MAIN_MODEL=qwen2.5-coder:3b \
AMADEUS_FAST_MODEL=qwen2.5-coder:1.5b \
AMADEUS_HEAVY_MODEL=qwen3:8b \
AMADEUS_OLLAMA_URL=http://localhost:11434 \
AMADEUS_RAG_DATA_DIR=.amadeus \
AMADEUS_RAG_TOP_K=8 \
AMADEUS_RAG_MARGIN=0.06 \
AMADEUS_RAG_SKIP_TESTS=1 \
AMADEUS_RAG_LEXICAL_WEIGHT=0.0 \
./amadeus.sh
```

`rag/rag.py query "your question" --debug` prints raw cosine distances to stderr, which is how the margin gets calibrated for a given codebase.

## What I learned

- Small local models (1.5B–3B) are genuinely useful for coding when tasks are scoped tightly — but not for grounded citation, where they silently ignore the instruction and fill gaps from training data.
- Prompt assembly, history windowing, and model selection are the real plumbing behind tools like Copilot. Building it by hand made that visible.
- Retrieval intuitions are usually wrong. Chunk boundaries looked like the obvious problem and weren't; test files looked like noise and were 44% of the corpus. An eval harness with a dozen verified question→file pairs settles in seconds what eyeballing output can't settle at all.
- A harness that reimplements the retrieval path instead of calling it will happily report good numbers for code that's broken in production. Ask me how I know.

## Roadmap

- [x] Local RAG over the codebase (sqlite-vec + code-trained embeddings)
- [x] Bundle the execution agent into the repo
- [x] Retrieval evaluation harness: question→expected-file pairs, recall@k and MRR
- [ ] Fix multi-concept query dilution (decomposition or model-based reranking)
- [ ] Grow the fixture set past 20 pairs so smaller effects are detectable
- [ ] Semantic session memory: embed history and retrieve relevant past exchanges instead of the blind last-40-lines window
- [ ] Streaming responses instead of waiting for full completion
- [ ] Configurable system prompt per project
