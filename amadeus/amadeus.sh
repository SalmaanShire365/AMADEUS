#!/bin/bash
# =============================================
# AMADEUS — Local AI Coding Agent with RAG
# Tiered local models via Ollama + retrieval
# over your codebase via sqlite-vec.
# No API keys, no cloud.
#
# Interactive:  amadeus
# One-shot CLI: amadeus index [dir]
#               amadeus rag [--debug] [--heavy] <question>
#               amadeus ask [--fast|--heavy] <task>
#               amadeus codex <task>
# =============================================

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# =========================
# MODELS (AMADEUS LAY3b)
# =========================
MAIN_MODEL="${AMADEUS_MAIN_MODEL:-qwen2.5-coder:3b}"
FAST_MODEL="${AMADEUS_FAST_MODEL:-qwen2.5-coder:1.5b}"
HEAVY_MODEL="${AMADEUS_HEAVY_MODEL:-qwen3:8b}"

# =========================
# COMPONENT PATHS
# =========================
RAG_PY="$SCRIPT_DIR/rag/rag.py"
RAG_VENV="$SCRIPT_DIR/rag/venv"
AGENT_PY="$SCRIPT_DIR/agent/agent.py"

# =========================
# HISTORY + SESSION
# =========================
HISTORY_FILE="$(pwd)/.agent_history.txt"
SESSION_ID=$(date +%s)

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT="You are AMADEUS, an expert software engineering agent.
You write clean, production-ready code. When asked to build something,
output complete files with no placeholders. Think step by step.
Be concise — avoid unnecessary explanation unless asked."

# =========================
# HELPERS
# =========================
rag_python() {
    if [[ -x "$RAG_VENV/bin/python3" ]]; then
        "$RAG_VENV/bin/python3" "$RAG_PY" "$@"
    else
        python3 "$RAG_PY" "$@"
    fi
}

# run_rag <model> <label> <debug:0|1> <question>
run_rag() {
    local MODEL="$1" LABEL="$2" DEBUG="$3" QUESTION="$4"

    if [[ "$DEBUG" == "1" ]]; then
        # Retrieval only: show distances, show what passed, skip the model.
        # This is the calibration tool for AMADEUS_RAG_THRESHOLD.
        rag_python query "$QUESTION" --debug > /dev/null
        return
    fi

    local GROUNDED_PROMPT
    GROUNDED_PROMPT=$(rag_python query "$QUESTION")
    if [[ "$GROUNDED_PROMPT" == "NO_RELEVANT_CONTEXT" ]]; then
        echo "Nothing relevant in the index for that question."
        return
    fi
    if [[ -z "$GROUNDED_PROMPT" ]]; then
        echo "Retrieval failed — see error above."
        return
    fi

    echo -e "\n--- $LABEL ($MODEL) ---"
    local RESPONSE
    RESPONSE=$(echo "$GROUNDED_PROMPT" | ollama run "$MODEL")
    echo "$RESPONSE"
    echo -e "------------------------\n"
    echo -e "[SESSION:$SESSION_ID] User: rag $QUESTION\nAgent: $RESPONSE\n---\n" >> "$HISTORY_FILE"
}

# run_model <model> <label> <task> <extra_context>
run_model() {
    local MODEL="$1" LABEL="$2" TASK="$3" EXTRA="$4"
    local RECENT_HISTORY=""
    [[ -f "$HISTORY_FILE" ]] && RECENT_HISTORY=$(tail -n 40 "$HISTORY_FILE")

    local PROMPT="${SYSTEM_PROMPT}

--- Recent History ---
$RECENT_HISTORY
--- End History ---
${EXTRA}

Task: $TASK"

    echo -e "\n--- $LABEL ($MODEL) ---"
    local RESPONSE
    RESPONSE=$(echo -e "$PROMPT" | ollama run "$MODEL")
    echo "$RESPONSE"
    echo -e "------------------------\n"
    echo -e "[SESSION:$SESSION_ID] User: $TASK\nAgent: $RESPONSE\n---\n" >> "$HISTORY_FILE"
}

usage() {
    cat << EOF
AMADEUS — local AI coding agent

Usage:
  amadeus                              interactive session
  amadeus index [dir]                  build/refresh RAG index (default: .)
  amadeus rag <question>               codebase-grounded answer
  amadeus rag --heavy <question>       grounded answer with $HEAVY_MODEL
  amadeus rag --debug <question>       retrieval only: distances, no model
  amadeus ask <task>                   one-shot query ($MAIN_MODEL)
  amadeus ask --fast <task>            one-shot with $FAST_MODEL
  amadeus ask --heavy <task>           one-shot with $HEAVY_MODEL
  amadeus codex <task>                 autonomous agent (writes files, runs code)
  amadeus help                         this message
EOF
}

# =========================
# ONE-SHOT CLI MODE
# =========================
if [[ $# -gt 0 ]]; then
    touch "$HISTORY_FILE"
    CMD="$1"; shift
    case "$CMD" in
        index)
            rag_python index "${1:-.}"
            ;;
        rag)
            MODEL=$HEAVY_MODEL; LABEL="AMADEUS-RAG"; DEBUG=0
            while [[ "$1" == --* ]]; do
                case "$1" in
                    --heavy) MODEL=$MAIN_MODEL; LABEL="AMADEUS-RAG-FAST" ;;
                    --debug) DEBUG=1 ;;
                    *) echo "unknown flag: $1"; exit 1 ;;
                esac
                shift
            done
            [[ $# -eq 0 ]] && { echo "usage: amadeus rag [--debug] [--heavy] <question>"; exit 1; }
            run_rag "$MODEL" "$LABEL" "$DEBUG" "$*"
            ;;
        ask)
            MODEL=$MAIN_MODEL; LABEL="AMADEUS"
            while [[ "$1" == --* ]]; do
                case "$1" in
                    --fast)  MODEL=$FAST_MODEL;  LABEL="AMADEUS-FAST" ;;
                    --heavy) MODEL=$HEAVY_MODEL; LABEL="AMADEUS-HEAVY" ;;
                    *) echo "unknown flag: $1"; exit 1 ;;
                esac
                shift
            done
            [[ $# -eq 0 ]] && { echo "usage: amadeus ask [--fast|--heavy] <task>"; exit 1; }
            run_model "$MODEL" "$LABEL" "$*" ""
            ;;
        codex)
            [[ $# -eq 0 ]] && { echo "usage: amadeus codex <task>"; exit 1; }
            python3 "$AGENT_PY" "$*" --model "$MAIN_MODEL"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            echo "unknown command: $CMD"
            usage
            exit 1
            ;;
    esac
    exit 0
fi

# =========================
# INTERACTIVE SESSION
# =========================
touch "$HISTORY_FILE"

echo "=== AMADEUS ==="
echo "Project: $(pwd)"
echo "Session: $SESSION_ID"
echo "Commands:"
echo "  file <path>        → include file in next task"
echo "  fast <task>        → quick model ($FAST_MODEL)"
echo "  heavy <task>       → strong model ($HEAVY_MODEL)"
echo "  index [dir]        → build/refresh RAG index (default: .)"
echo "  rag <question>     → codebase-grounded answer"
echo "  rag heavy <quest.> → grounded answer with $HEAVY_MODEL"
echo "  rag debug <quest.> → retrieval only: distances, no model"
echo "  codex <task>       → autonomous agent (writes files, runs code)"
echo "  shell <cmd>        → run a shell command"
echo "  history            → show session history"
echo "  clear              → clear history file"
echo "  exit               → quit"
echo ""

FILE_CONTEXT=""

while true; do
    read -p "Task> " TASK

    if [[ "$TASK" == "exit" ]]; then
        echo "Leaving timeline..."
        break
    fi

    if [[ "$TASK" == "history" ]]; then
        cat "$HISTORY_FILE"
        continue
    fi

    if [[ "$TASK" == "clear" ]]; then
        > "$HISTORY_FILE"
        FILE_CONTEXT=""
        echo "History cleared."
        continue
    fi

    if [[ "$TASK" == shell* ]]; then
        REAL_TASK="${TASK#shell }"
        echo -e "\n--- SHELL EXECUTION ---"
        eval "$REAL_TASK"
        echo -e "----------------------\n"
        echo -e "[SESSION:$SESSION_ID] User: shell $REAL_TASK\nAgent: [shell executed]\n" >> "$HISTORY_FILE"
        continue
    fi

    if [[ "$TASK" == index* ]]; then
        TARGET="${TASK#index}"
        TARGET="${TARGET# }"
        echo -e "\n--- INDEXING ${TARGET:-.} ---"
        rag_python index "${TARGET:-.}"
        echo -e "-----------------------------\n"
        continue
    fi

    if [[ "$TASK" == rag* ]]; then
        QUESTION="${TASK#rag }"
        MODEL=$MAIN_MODEL; LABEL="AMADEUS-RAG"; DEBUG=0
        if [[ "$QUESTION" == heavy* ]]; then
            QUESTION="${QUESTION#heavy }"
            MODEL=$HEAVY_MODEL; LABEL="AMADEUS-RAG-HEAVY"
        elif [[ "$QUESTION" == debug* ]]; then
            QUESTION="${QUESTION#debug }"
            DEBUG=1
        fi
        run_rag "$MODEL" "$LABEL" "$DEBUG" "$QUESTION"
        continue
    fi

    if [[ "$TASK" == codex* ]]; then
        REAL_TASK="${TASK#codex }"
        echo -e "\n--- LOCAL AGENT EXECUTION ---"
        python3 "$AGENT_PY" "$REAL_TASK" --model "$MAIN_MODEL"
        echo -e "-----------------------------\n"
        echo -e "[SESSION:$SESSION_ID] User: codex $REAL_TASK\nAgent: [local-agent executed]\n" >> "$HISTORY_FILE"
        continue
    fi

    if [[ "$TASK" == file* ]]; then
        FILE_PATH="${TASK#file }"
        if [[ -f "$FILE_PATH" ]]; then
            FILE_CONTENT=$(cat "$FILE_PATH")
            FILE_CONTEXT="${FILE_CONTEXT}\n\n### File: $FILE_PATH\n\`\`\`\n$FILE_CONTENT\n\`\`\`"
            echo "Loaded: $FILE_PATH (will be included in next task)"
        else
            echo "File not found: $FILE_PATH"
        fi
        continue
    fi

    MODEL=$MAIN_MODEL
    LABEL="AMADEUS"
    if [[ "$TASK" == fast* ]]; then
        MODEL=$FAST_MODEL
        TASK="${TASK#fast }"
        LABEL="AMADEUS-FAST"
    elif [[ "$TASK" == heavy* ]]; then
        MODEL=$HEAVY_MODEL
        TASK="${TASK#heavy }"
        LABEL="AMADEUS-HEAVY"
    fi

    run_model "$MODEL" "$LABEL" "$TASK" "$FILE_CONTEXT"
    FILE_CONTEXT=""
done
