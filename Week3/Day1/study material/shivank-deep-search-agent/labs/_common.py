"""Small helpers shared by the labs (printing, paths). Import this FIRST in every lab."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # lets `python labs/labXX.py` import the deepsearch package

try:  # make the box-drawing characters print on Windows terminals too
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def step(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(3, 56 - len(title)))


def wrap(text: str, width: int = 96, indent: str = "  ") -> str:
    return textwrap.fill(str(text), width=width, initial_indent=indent, subsequent_indent=indent)


def table(rows: list[dict], columns: list[str] | None = None, max_width: int = 38) -> None:
    """Print a list of dicts as a simple aligned text table."""
    if not rows:
        print("  (no rows)")
        return
    columns = columns or list(rows[0].keys())

    def cell(v) -> str:
        s = "" if v is None else str(v)
        return s if len(s) <= max_width else s[: max_width - 1] + "…"

    widths = {c: max(len(c), *(len(cell(r.get(c))) for r in rows)) for c in columns}
    print("  " + " | ".join(c.ljust(widths[c]) for c in columns))
    print("  " + "-+-".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  " + " | ".join(cell(r.get(c)).ljust(widths[c]) for c in columns))


def pause_note(text: str) -> None:
    print("\n" + wrap("👉 " + text))
