# AMADEUS — common tasks. Run from the repo root: `make help`.

APP    := amadeus
VENV   := $(APP)/venv
PY     := $(abspath $(VENV))/bin/python
TUI    := $(APP)/tui

# Eval index (built from the gastown corpus, tests skipped) and its source.
DATA   ?= $(or $(AMADEUS_RAG_DATA_DIR),$(HOME)/.amadeus-notest)
CORPUS ?= $(HOME)/rag-corpus/gastown/internal

# `make demo` runs on a fresh copy of demo/workspace so every rehearsal starts
# clean and the repo never collects demo output.
DEMO_DIR ?= $(or $(TMPDIR),/tmp)/amadeus-demo

.DEFAULT_GOAL := help
.PHONY: help setup models demo eval task-eval index tui test lint check

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

setup: ## create the venv, install Python deps, fetch Rust deps
	test -x $(PY) || python3 -m venv $(VENV)
	$(PY) -m pip install -q -r $(APP)/rag/requirements.txt
	cd $(TUI) && cargo fetch

models: ## pull the default main + fast models (about 8 GB, once)
	ollama pull $(or $(AMADEUS_MAIN_MODEL),qwen3:8b)
	ollama pull $(or $(AMADEUS_FAST_MODEL),qwen3:4b-instruct)

demo: ## build, run the doctor, open the TUI on a fresh copy of demo/workspace
	cd $(TUI) && cargo build --release --quiet
	$(APP)/amadeus.sh doctor
	@if [ -e "$(DEMO_DIR)" ]; then \
		grep -qs "Quill & Kettle" "$(DEMO_DIR)/README.txt" || \
			{ echo "$(DEMO_DIR) exists and isn't a demo copy; refusing to replace it (set DEMO_DIR)"; exit 1; }; \
		rm -rf "$(DEMO_DIR)"; \
	fi
	cp -R demo/workspace "$(DEMO_DIR)"
	cd "$(DEMO_DIR)" && $(abspath $(APP))/amadeus.sh

eval: ## retrieval eval (recall@k, MRR) against $(DATA)
	@test -f $(DATA)/index.db || { echo "no index at $(DATA) — run: make index"; exit 1; }
	@! test -e $(DATA)/index.db-journal || { echo "$(DATA) has a leftover index.db-journal (interrupted write). Opening it would roll it back; refusing."; exit 1; }
	cd $(APP) && AMADEUS_RAG_DATA_DIR=$(DATA) $(PY) rag/eval/run_eval.py

task-eval: ## score the templates on eval/workspace with a real model (slow; read amadeus/eval/README.md first)
	cd $(APP)/eval && $(PY) run_eval.py $(ARGS)

index: ## (re)build the eval index: CORPUS -> DATA
	cd $(APP) && AMADEUS_RAG_DATA_DIR=$(DATA) $(PY) rag/rag.py index $(CORPUS)

tui: ## build and run the TUI (release)
	cd $(TUI) && cargo run --release

test: ## Rust tests, Python byte-compile, task eval scorer and harness tests (no model)
	cd $(TUI) && cargo test
	$(PY) -m py_compile $(APP)/rag/rag.py $(APP)/rag/eval/run_eval.py
	cd $(TUI) && cargo build --release --quiet
	cd $(APP)/eval && $(PY) -B test_scoring.py && $(PY) -B test_harness.py

lint: ## rustfmt check + clippy (warnings are errors)
	cd $(TUI) && cargo fmt --check
	cd $(TUI) && cargo clippy --all-targets -- -D warnings

check: lint test eval ## everything: lint, test, eval
