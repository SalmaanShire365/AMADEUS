#!/usr/bin/env python3
"""
run_eval.py — retrieval eval harness for rag.py.

Loads rag/eval/fixtures.jsonl, runs each query through rag.py's own
embed() + vec_chunks retrieval (same embedding model, same SQLite/
sqlite-vec index), and reports recall@k / MRR at the file_path level.

Run from the directory that contains the "rag" package and the built
".amadeus" index (i.e. wherever you ran `rag.py index`):

    python rag/eval/run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402  (path must be set up first)

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures.jsonl"
K_VALUES = [1, 3, 5, 10]
RETRIEVE_K = max(K_VALUES)


def load_fixtures(path: Path) -> list[dict]:
    fixtures = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                fixtures.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"error: {path}:{lineno}: invalid JSON ({e})")
    return fixtures


def retrieve_ranked_docs(question: str) -> list[str]:
    """Run the same embed + vec_chunks KNN search rag.py's cmd_query uses,
    without the CLI-only distance threshold / prompt formatting, and
    collapse to a ranked list of unique file_paths (best rank per file)."""
    qvec = rag.serialize_f32(rag.embed(question, is_query=True))
    rows = db.execute(
        """SELECT c.file_path, v.distance
           FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id
           WHERE v.embedding MATCH ? AND k = ?
           ORDER BY v.distance""",
        (qvec, RETRIEVE_K),
    ).fetchall()

    ranked_docs = []
    seen = set()
    for file_path, _dist in rows:
        if file_path not in seen:
            seen.add(file_path)
            ranked_docs.append(file_path)
    return ranked_docs


def recall_at_k(expected: list[str], ranked_docs: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top_k = set(ranked_docs[:k])
    hit = len(set(expected) & top_k)
    return hit / len(expected)


def reciprocal_rank(expected: list[str], ranked_docs: list[str]) -> float:
    expected_set = set(expected)
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in expected_set:
            return 1.0 / rank
    return 0.0


def main() -> None:
    if not rag.DB_PATH.exists():
        sys.exit(
            f"error: no index found at {rag.DB_PATH.resolve()}. "
            f"Run `python rag/rag.py index <dir>` first."
        )

    fixtures = load_fixtures(FIXTURES_PATH)

    global db
    db = rag.open_db()

    results = []
    for fx in fixtures:
        query = fx["query"]
        expected = fx["expected"]
        ranked_docs = retrieve_ranked_docs(query)
        row = {
            "query": query,
            "recall": {k: recall_at_k(expected, ranked_docs, k) for k in K_VALUES},
            "rr": reciprocal_rank(expected, ranked_docs),
        }
        results.append(row)

    header = ["query"] + [f"recall@{k}" for k in K_VALUES] + ["RR"]
    col_widths = [40] + [10] * len(K_VALUES) + [6]
    print("  ".join(h.ljust(w) for h, w in zip(header, col_widths)))
    print("  ".join("-" * w for w in col_widths))
    for row in results:
        q = row["query"][:37] + "..." if len(row["query"]) > 40 else row["query"]
        cells = [q.ljust(col_widths[0])]
        cells += [f"{row['recall'][k]:.2f}".ljust(w) for k, w in zip(K_VALUES, col_widths[1:-1])]
        cells.append(f"{row['rr']:.2f}".ljust(col_widths[-1]))
        print("  ".join(cells))

    n = len(results)
    avg_recall = {k: sum(r["recall"][k] for r in results) / n for k in K_VALUES}
    avg_rr = sum(r["rr"] for r in results) / n
    summary = " ".join(f"recall@{k}={avg_recall[k]:.3f}" for k in K_VALUES)
    print(f"\nSUMMARY ({n} queries): {summary} MRR={avg_rr:.3f}")


if __name__ == "__main__":
    main()
