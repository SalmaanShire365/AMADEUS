# AMADEUS — common tasks. Run from the repo root: `make help`.

APP    := amadeus
VENV   := $(APP)/venv
PY     := $(abspath $(VENV))/bin/python
TUI    := $(APP)/tui

# Eval index (built from the gastown corpus, tests skipped) and its source.
DATA   ?= $(or $(AMADEUS_RAG_DATA_DIR),$(HOME)/.amadeus-notest)
CORPUS ?= $(HOME)/rag-corpus/gastown/internal

.DEFAULT_GOAL := help
.PHONY: help setup eval index tui test lint check

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-7s %s\n", $$1, $$2}'

setup: ## create the venv, install Python deps, fetch Rust deps
	test -x $(PY) || python3 -m venv $(VENV)
	$(PY) -m pip install -q -r $(APP)/rag/requirements.txt
	cd $(TUI) && cargo fetch

eval: ## retrieval eval (recall@k, MRR) against $(DATA)
	@test -f $(DATA)/index.db || { echo "no index at $(DATA) — run: make index"; exit 1; }
	@! test -e $(DATA)/index.db-journal || { echo "$(DATA) has a leftover index.db-journal (interrupted write). Opening it would roll it back; refusing."; exit 1; }
	cd $(APP) && AMADEUS_RAG_DATA_DIR=$(DATA) $(PY) rag/eval/run_eval.py

index: ## (re)build the eval index: CORPUS -> DATA
	cd $(APP) && AMADEUS_RAG_DATA_DIR=$(DATA) $(PY) rag/rag.py index $(CORPUS)

tui: ## build and run the TUI (release)
	cd $(TUI) && cargo run --release

test: ## Rust tests + Python byte-compile check
	cd $(TUI) && cargo test
	$(PY) -m py_compile $(APP)/rag/rag.py $(APP)/rag/eval/run_eval.py $(APP)/agent/agent.py

lint: ## rustfmt check + clippy (warnings are errors)
	cd $(TUI) && cargo fmt --check
	cd $(TUI) && cargo clippy --all-targets -- -D warnings

check: lint test eval ## everything: lint, test, eval
