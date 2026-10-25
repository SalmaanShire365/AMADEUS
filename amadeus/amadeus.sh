#!/usr/bin/env bash
# =============================================
# AMADEUS — local AI task harness on Ollama.
# No API keys, no cloud.
#
# AMADEUS_HOME is always the directory this script lives in (symlinks
# resolved). It's exported so the TUI and rag.py agree on the checkout; the
# TUI applies the same rule to its own executable path.
# =============================================

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export AMADEUS_HOME="$SCRIPT_DIR"

RAG_PY="$SCRIPT_DIR/rag/rag.py"
VENV_PY="$SCRIPT_DIR/venv/bin/python3"
AGENT_PY="$SCRIPT_DIR/agent/agent.py"
TUI_DIR="$SCRIPT_DIR/tui"
TUI_BIN="$TUI_DIR/target/release/amadeus-tui"

MAIN_MODEL="${AMADEUS_MAIN_MODEL:-qwen3:8b}"

python_bin() {
    if [[ -x "$VENV_PY" ]]; then echo "$VENV_PY"; else echo python3; fi
}

rag_python() {
    "$(python_bin)" "$RAG_PY" "$@"
}

# Build the TUI when it's missing or older than any of its sources, so a
# stale binary never runs after a code change.
ensure_tui() {
    local reason=""
    if [[ ! -x "$TUI_BIN" ]]; then
        reason="not built yet"
    elif [[ -n "$(find "$TUI_DIR/src" "$TUI_DIR/Cargo.toml" "$TUI_DIR/Cargo.lock" \
                    -newer "$TUI_BIN" -print -quit 2>/dev/null)" ]]; then
        reason="sources changed"
    fi
    [[ -z "$reason" ]] && return 0

    if ! command -v cargo > /dev/null; then
        echo "The AMADEUS TUI needs Rust to build ($reason)." >&2
        echo "Install it from https://rustup.rs, then run this again." >&2
        exit 1
    fi
    echo "Building the AMADEUS TUI ($reason)..." >&2
    (cd "$TUI_DIR" && cargo build --release --quiet) || {
        echo "Build failed. Run: cd $TUI_DIR && cargo build --release" >&2
        exit 1
    }
    # cargo may decide nothing needed relinking; mark the binary current so
    # the staleness check doesn't fire on every launch.
    touch "$TUI_BIN"
}

usage() {
    cat << EOF
AMADEUS — local AI task harness (Ollama, nothing leaves this machine)

Usage:
  amadeus                         open the TUI in the current directory
  amadeus tui                     same
  amadeus tui-dev                 run the TUI with cargo run (debug build)
  amadeus ask [--fast|--heavy] <task>
                                  one-shot answer, streamed to stdout
  amadeus rag [--debug] [--heavy] <question>
                                  one-shot answer grounded in the RAG index
  amadeus index [dir]             build or refresh the RAG index
  amadeus agent <task>            coding agent (alias: codex)
  amadeus help                    this message
EOF
}

CMD="${1:-tui}"
[[ $# -gt 0 ]] && shift

case "$CMD" in
    tui)
        ensure_tui
        exec "$TUI_BIN" "$@"
        ;;
    tui-dev)
        cd "$TUI_DIR" || exit 1
        exec cargo run -- "$@"
        ;;
    ask|rag)
        ensure_tui
        exec "$TUI_BIN" "$CMD" "$@"
        ;;
    index)
        rag_python index "${1:-.}"
        ;;
    agent|codex)
        [[ $# -eq 0 ]] && { echo "usage: amadeus agent <task>"; exit 1; }
        exec "$(python_bin)" "$AGENT_PY" "$*" --model "$MAIN_MODEL"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "unknown command: $CMD" >&2
        usage >&2
        exit 1
        ;;
esac
