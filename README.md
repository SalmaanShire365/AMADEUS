# AMADEUS

A fully local, tiered AI coding assistant in ~150 lines of bash. No API keys, no subscriptions, no cloud. Every model runs on my own hardware through [Ollama](https://ollama.com).

AMADEUS is an interactive terminal session that routes each task to a different local model depending on difficulty, keeps rolling session memory, lets you inject files into the prompt, and can hand off to a separate autonomous Python agent when a task needs real code execution instead of just code generation.

```
=== AMADEUS + CODEX HYBRID SYSTEM ===
Project: /home/shire
Session: 1783372494
Commands:
  file <path>       → include file in next task
  fast <task>       → quick model (qwen2.5-coder:1.5b)
  heavy <task>      → strong model (qwen3:8b)
  codex <task>      → real code execution via local agent
  shell <cmd>       → run a shell command
  history           → show session history
  clear             → clear history file
  exit              → quit
```

## Why local, and why these models

This project was built under a real constraint: my development machine is a convertible laptop with no dedicated GPU and 8GB of RAM. That rules out running large local models, and I didn't want to depend on paid cloud APIs for everyday coding help.

**My machine:**

| Component | Spec |
|---|---|
| Machine | HP Pavilion x360 Convertible 14m-dw0xxx |
| OS | Arch Linux x86_64 |
| CPU | Intel Core i5-1035G1 (8 threads) @ 3.60 GHz |
| GPU | Intel Iris Plus G1 (integrated) — Ollama runs CPU-only |
| RAM | 8GB (7.44 GiB usable, shared with the iGPU) |
| Storage | 256GB SSD |

With ~7.4 GiB of usable RAM shared between the OS, my dev tools, and the model, an 8B model at Q4 quantization (~5GB) is the practical ceiling — and even that leaves little headroom, so it's reserved for tasks that actually need it.

The constraint shaped the design. Instead of one big model, AMADEUS uses **tiered routing**: send each task to the smallest model that can handle it.

| Tier | Model | Size | Used for |
|---|---|---|---|
| `fast` | qwen2.5-coder:1.5b | ~1GB | Quick lookups, one-liners, syntax questions |
| default | qwen2.5-coder:3b | ~2GB | Everyday coding tasks |
| `heavy` | qwen3:8b | ~5GB | Harder reasoning, architecture questions |

On CPU-only hardware, the difference between a 1.5B and an 8B model is the difference between a near-instant answer and a long wait. Routing by difficulty keeps the tool usable all day instead of only when I'm willing to wait. It also turned out to be a useful lesson in its own right: small models are far more capable than expected when the task is scoped correctly.

## Architecture

```
Task> input
   │
   ├── file <path>   → stack file contents into next prompt
   ├── shell <cmd>   → passthrough to the shell
   ├── codex <task>  → hand off to autonomous local-agent (plan → write → run → review)
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
- **Execution handoff.** `codex <task>` calls a separate Python agent that can actually write files, run commands, and review its own output — the generation/execution split keeps this script simple.

## RAG: codebase-grounded answers

AMADEUS can index a repository and answer questions grounded only in that code — a fully local RAG pipeline with zero API dependencies.

```
Task> index .
indexed: src/ingest.py
indexed: README.md
done: 42 chunks added, 0 files unchanged

Task> rag how does the payment retry logic work
--- AMADEUS-RAG (qwen2.5-coder:3b) ---
The retry logic lives in src/payments.py (lines 88-120)...
```

How it works: `rag/rag.py` chunks source files on function/class boundaries (not fixed windows — slicing a function in half ruins retrieval), embeds each chunk with `nomic-embed-text` through Ollama's API, and stores vectors in a single SQLite file (`.amadeus/index.db`) using sqlite-vec — no vector database server eating RAM. Queries embed the question, pull the top-k nearest chunks by cosine distance, and inject them into a grounded prompt.

Design decisions worth noting:

- **Grounding is two layers.** Naive vector search always returns *something*, even for unrelated questions. A cosine distance threshold rejects weak matches (the tool answers "nothing relevant in the index" instead of hallucinating), and the prompt instructs the model to answer only from context and cite file:line.
- **Location is baked into the vectors.** Every chunk is embedded with a `# File: path, lines N-M` header, so questions that mention a module name match its chunks even when the code never uses that word.
- **Incremental re-indexing.** File hashes in `.amadeus/index_meta.json` mean unchanged files are skipped and deleted files are purged — re-indexing after an edit takes seconds.
- **Memory sequencing.** Retrieval runs and exits before the chat model is invoked, so the embedder and an 8B model never occupy RAM simultaneously — necessary on 8GB.

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
   ollama pull nomic-embed-text
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
│   ├── rag.py          # chunk, embed, store, retrieve
│   └── requirements.txt
├── agent/
│   └── agent.py        # autonomous plan → write → run → review loop
└── .amadeus/           # per-project index (created at runtime, gitignored)
```

### Configuration

Everything is overridable with environment variables:

```bash
AMADEUS_MAIN_MODEL=qwen2.5-coder:3b \
AMADEUS_FAST_MODEL=qwen2.5-coder:1.5b \
AMADEUS_HEAVY_MODEL=qwen3:8b \
AMADEUS_EMBED_MODEL=nomic-embed-text \
AMADEUS_RAG_TOP_K=5 \
AMADEUS_RAG_THRESHOLD=0.62 \
./amadeus.sh
```

Tune the threshold for your codebase: `rag/venv/bin/python3 rag/rag.py query "your question" --debug` prints the raw cosine distances so you can see where relevant and irrelevant chunks separate.

## What I learned

- Small local models (1.5B–3B) are genuinely useful for coding when tasks are scoped tightly.
- Prompt assembly, history windowing, and model selection are the real plumbing behind tools like Copilot — building it by hand made that visible.
- Hardware constraints are a design input, not just a limitation. Tiered routing exists in this project because my machine forced the question.

## Roadmap

- [x] Local RAG over the codebase (sqlite-vec + nomic-embed-text)
- [x] Bundle the execution agent into the repo
- [ ] Semantic session memory: embed history and retrieve relevant past exchanges instead of the blind last-40-lines window
- [ ] Streaming responses instead of waiting for full completion
- [ ] Configurable system prompt per project
- [ ] Retrieval evaluation harness: a small set of question→expected-file pairs to measure chunking changes