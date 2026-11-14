"""Small helpers for printing readable output in the labs."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

try:  # make ₹ and Hindi print correctly on Windows terminals
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass


def banner(title: str) -> None:
    line = "=" * min(88, max(40, len(title) + 8))
    print(f"\n{line}\n  {title}\n{line}")


def step(text: str) -> None:
    print(f"\n--> {text}")


def table(rows: list[dict], columns: list[str] | None = None, max_width: int = 60) -> None:
    """Print a list of dicts as an aligned text table."""
    if not rows:
        print("(no rows)")
        return
    columns = columns or list(rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        s = "" if v is None else str(v)
        s = " ".join(s.split())
        return s if len(s) <= max_width else s[: max_width - 3] + "..."

    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


@contextmanager
def timer(label: str = "", out: dict | None = None):
    t0 = time.perf_counter()
    yield
    ms = (time.perf_counter() - t0) * 1000
    if out is not None:
        out[label] = ms
    elif label:
        print(f"   [{label}: {ms:.1f} ms]")


def show_hits(hits, n: int = 5, width: int = 70) -> None:
    rows = []
    for i, h in enumerate(hits[:n], start=1):
        rows.append({"#": i, "doc": h.chunk.doc_id, "score": h.score,
                     "cos": h.vector_score, "bm25": h.bm25_score,
                     "rerank": h.rerank_score, "text": h.chunk.short(width)})
    cols = ["#", "doc", "score", "cos", "bm25"] + (["rerank"] if any(r["rerank"] is not None for r in rows) else []) + ["text"]
    table(rows, cols, max_width=width)
