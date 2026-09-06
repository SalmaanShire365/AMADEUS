#!/usr/bin/env python3
"""
rag.py — local RAG engine for AMADEUS.

Two subcommands:
  index <dir>     chunk + embed + store every source file under <dir>
  query <text>    embed the question, retrieve top-k chunks, print a
                  grounded prompt on stdout (bash pipes it to ollama)

Everything is local: embeddings come from Ollama (nomic-embed-text),
vectors live in a single SQLite file via sqlite-vec.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import urllib.request
from pathlib import Path

# =========================
# CONFIG
# =========================
OLLAMA_URL = os.environ.get("AMADEUS_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "unclemusclez/jina-embeddings-v2-base-code"
USE_TASK_PREFIX = False
SKIP_TESTS = os.environ.get("AMADEUS_RAG_SKIP_TESTS","1") == "1"
EMBED_DIM = 768
TOP_K = int(os.environ.get("AMADEUS_RAG_TOP_K", "8"))
LEXICAL_WEIGHT = float(os.environ.get("AMADEUS_RAG_LEXICAL_WEIGHT", "0.0"))
# Relative cutoff: keep hits within DISTANCE_MARGIN of the closest match.
# Absolute thresholds don't work here — distance bands shift per query
# (0.69-0.78 on one, 0.49-0.59 on another), and an off-topic question
# can score better than a real one's tail. Width is tunable; relevance
# is not filterable at this layer.
DISTANCE_THRESHOLD = float(os.environ.get("AMADEUS_RAG_THRESHOLD", "0.78"))
DISTANCE_MARGIN = float(os.environ.get("AMADEUS_RAG_MARGIN", "0.06"))
MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200

DATA_DIR = Path(os.environ.get("AMADEUS_RAG_DATA_DIR", ".amadeus")).resolve()
DB_PATH = DATA_DIR / "index.db"
META_PATH = DATA_DIR / "index_meta.json"

IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "uv.lock",
    ".agent_history.txt", ".ollama_agent_history.txt",
}

CODE_EXTS = {
    ".py", ".sh", ".bash", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".lua",
}
TEXT_EXTS = CODE_EXTS | {
    ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".sql",
    ".html", ".css", ".cfg", ".ini", ".env.example",
}


IGNORE_DIRS = {
    ".git", ".amadeus", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", "target", ".idea", ".vscode", ".cache", ".claude",
    "eval",
}



# Top-level definition boundaries across the languages I actually use.
DEF_RE = re.compile(
    r"^(def |class |async def |function |fn |func |pub fn |impl |"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{)",
    re.MULTILINE,
)


# =========================
# OLLAMA EMBEDDINGS
# =========================
def embed(text: str, is_query: bool = False) -> list[float]:
    """Embed text via Ollama. nomic-embed-text is trained with task
    prefixes — using them measurably improves retrieval."""
    prefix = ("search_query: " if is_query else "search_document: ") if USE_TASK_PREFIX else ""
    payload = json.dumps(
        {"model": EMBED_MODEL, "prompt": prefix + text}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["embedding"]
    except urllib.error.URLError as e:  # type: ignore
        sys.exit(f"error: cannot reach Ollama at {OLLAMA_URL} ({e}). "
                 f"Is `ollama serve` running? Did you `ollama pull {EMBED_MODEL}`?")


def serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# =========================
# DATABASE
# =========================
def open_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    try:
        import sqlite_vec  # type: ignore
        sqlite_vec.load(db)
    except ImportError:
        sys.exit("error: sqlite-vec not installed. Run: "
                 "pip install sqlite-vec (inside rag/venv)")
    db.enable_load_extension(False)
    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            content TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER,
            embedding FLOAT[{EMBED_DIM}] distance_metric=cosine
        );
    """)
    return db


def load_meta() -> dict:
    if META_PATH.exists() and DB_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"files": {}}


def save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2))


# =========================
# CHUNKING
# =========================
def split_fixed(text: str, start_line: int) -> list[tuple[int, int, str]]:
    """Fallback: line-aligned chunks with line overlap. Returns (start, end, text)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    out = []
    i = 0
    while i < len(lines):
        piece_lines, size, j = [], 0, i
        while j < len(lines) and (not piece_lines or size + len(lines[j]) <= MAX_CHUNK_CHARS):
            piece_lines.append(lines[j])
            size += len(lines[j])
            j += 1
        piece = "".join(piece_lines)
        if piece.strip():
            out.append((start_line + i, start_line + j - 1, piece))
        if j >= len(lines):
            break
        # step back a few lines for overlap, but always advance
        overlap = 0
        back = 0
        while back < len(piece_lines) - 1 and overlap + len(piece_lines[-1 - back]) <= OVERLAP_CHARS:
            overlap += len(piece_lines[-1 - back])
            back += 1
        i = max(i + 1, j - back)
    return out


def chunk_file(path: Path) -> list[tuple[int, int, str]]:
    """Structure-aware chunking. Code splits on definition boundaries,
    markdown on headings, everything else on fixed windows."""
    text = path.read_text(errors="replace")
    if not text.strip():
        return []

    if path.suffix in CODE_EXTS:
        boundaries = [m.start() for m in DEF_RE.finditer(text)]
        if not boundaries:
            return split_fixed(text, 1)
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(len(text))
        chunks = []
        for a, b in zip(boundaries, boundaries[1:]):
            piece = text[a:b]
            start = text[:a].count("\n") + 1
            if len(piece) > MAX_CHUNK_CHARS:
                chunks.extend(split_fixed(piece, start))
            elif piece.strip():
                chunks.append((start, start + piece.count("\n"), piece))
        return chunks

    if path.suffix == ".md":
        parts = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
        chunks, line = [], 1
        for part in parts:
            if part.strip():
                if len(part) > MAX_CHUNK_CHARS:
                    chunks.extend(split_fixed(part, line))
                else:
                    chunks.append((line, line + part.count("\n"), part))
            line += part.count("\n")
        return chunks

    return split_fixed(text, 1)


# =========================
# INDEX
# =========================
def cmd_index(root: str) -> None:
    root_path = Path(root).resolve()
    db_existed = DB_PATH.exists()
    db = open_db()
    meta = load_meta() if db_existed else {"files" : {}}
    seen, added, skipped = set(), 0, 0
    print(f"index: {DB_PATH}")
    for path in sorted(root_path.rglob("*")):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in TEXT_EXTS:
            continue
        if path.name in IGNORE_FILES or path.name.startswith(".agent"):
            continue
        
        if SKIP_TESTS and path.name.endswith(("_test.go", "_test.py", ".test.ts", ".spec.ts")):
            continue


        rel = str(path.relative_to(root_path))
        seen.add(rel)
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()

        if meta["files"].get(rel) == file_hash:
            skipped += 1
            continue

        # File changed (or is new): drop its old chunks, re-embed.
        old_ids = [r[0] for r in db.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (rel,))]
        if old_ids:
            qmarks = ",".join("?" * len(old_ids))
            db.execute(f"DELETE FROM chunks WHERE id IN ({qmarks})", old_ids)
            db.execute(
                f"DELETE FROM vec_chunks WHERE chunk_id IN ({qmarks})", old_ids)

        for start, end, content in chunk_file(path):
            # Bake location into the embedded text so questions that
            # mention a file or module name match its chunks.
            headed = f"# File: {rel}, lines {start}-{end}\n{content}"
            cur = db.execute(
                "INSERT INTO chunks (file_path, start_line, end_line, content) "
                "VALUES (?, ?, ?, ?)",
                (rel, start, end, content),
            )
            db.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, serialize_f32(embed(headed))),
            )
            added += 1
        meta["files"][rel] = file_hash
        print(f"indexed: {rel}")

    # Purge files that were deleted from disk.
    for rel in [f for f in meta["files"] if f not in seen]:
        old_ids = [r[0] for r in db.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (rel,))]
        if old_ids:
            qmarks = ",".join("?" * len(old_ids))
            db.execute(f"DELETE FROM chunks WHERE id IN ({qmarks})", old_ids)
            db.execute(
                f"DELETE FROM vec_chunks WHERE chunk_id IN ({qmarks})", old_ids)
        del meta["files"][rel]
        print(f"removed: {rel}")

    db.commit()
    save_meta(meta)
    print(f"\ndone: {added} chunks added, {skipped} files unchanged")


# =========================
# QUERY
# =========================
PROMPT_TEMPLATE = """You are AMADEUS. Answer using ONLY the code context below.
If the context does not contain the answer, say "Not found in the indexed code" — do not guess.
Begin every bullet or paragraph with the source in square brackets,
e.g. [ml/predict.py:34-80]. Never state a fact without a source tag.

--- Context ---
{context}
--- End context ---

Question: {question}"""


STOPWORDS = {"the", "a", "an", "how", "does", "do", "what", "is", "are",
             "in", "to", "of", "when", "where", "with", "work", "works"}

def lexical_score(query: str, file_path: str, content: str) -> float:
    terms = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
    if not terms:
        return 0.0
    hay = (file_path + " " + content).lower()
    return sum(1 for t in terms if t in hay) / len(terms)

def rerank(question: str, rows: list) -> list:
    """Blend vector distance with query-term overlap.
    rows must end with (..., content, distance)."""
    return sorted(rows, key=lambda r: r[-1] - LEXICAL_WEIGHT * lexical_score(question, r[0], r[-2]))

def cmd_query(question: str, debug: bool = False) -> None:
    if not DB_PATH.exists():
        sys.exit(f"error: no index found at {DB_PATH}. Run `index <dir>` first.")
    db = open_db()
    qvec = serialize_f32(embed(question, is_query=True))
    
    rows = rerank(question,rows)
        
    if debug:
        for fp, s, e, _, dist in rows:
            print(f"  {dist:.4f}  {fp}:{s}-{e}", file=sys.stderr)
    
    rows = sorted(rows, key=lambda r: r[4] - LEXICAL_WEIGHT * lexical_score(question, r[0], r[3]))

    if not rows:
        print("NO_RELEVANT_CONTEXT")
        return
    
    best = rows[0][4]
    rows = [r for r in rows if r[4] <= best + DISTANCE_MARGIN]
   

    context = "\n\n".join(
        f"### {fp} (lines {s}-{e})\n```\n{content}\n```"
        for fp, s, e, content, _ in rows
    )
    print(PROMPT_TEMPLATE.format(context=context, question=question))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMADEUS local RAG engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index a directory")
    p_index.add_argument("dir", nargs="?", default=".")

    p_query = sub.add_parser("query", help="retrieve grounded context")
    p_query.add_argument("question")
    p_query.add_argument("--debug", action="store_true",
                         help="print retrieval distances to stderr")

    args = parser.parse_args()
    if args.cmd == "index":
        cmd_index(args.dir)
    else:
        cmd_query(args.question, args.debug)
