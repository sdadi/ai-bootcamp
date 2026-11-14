"""
tracing.py — a trace of every step, and a budget that can stop the run.

From the Deep Search course: Module 9 (Cost, latency and budgets) and the `tracing.py` / `budget.py`
pair in the code pack. Production RAG course Module 12 ("Observability") covers the same idea for RAG.

Why both exist:
* A trace (one JSON line per event) is how you debug a wrong verdict: you read backwards from the verdict
  to the evidence to the page. The report alone never tells you WHY.
* A budget (tokens, USD, wall-clock) is how you stop an agent from spending without limit when the web
  is slow or a model loops.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RUNS_DIR, SETTINGS


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token). Replace with tiktoken for exact counts (Production RAG Lab 13)."""
    return max(1, len(text) // 4)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_usd: float = field(default_factory=lambda: SETTINGS.max_usd)
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    started: float = field(default_factory=time.time)

    def charge(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.spent_usd += (input_tokens * SETTINGS.price_in_per_mtok + output_tokens * SETTINGS.price_out_per_mtok) / 1_000_000
        if self.spent_usd > self.max_usd:
            raise BudgetExceeded(f"Budget of ${self.max_usd:.2f} exceeded (spent ${self.spent_usd:.3f}). Raise VRA_MAX_USD or reduce pages.")

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started


class Tracer:
    """Writes runs/<timestamp>-<name>/trace.jsonl. Use `t.event(stage, **data)` liberally."""

    def __init__(self, name: str = "assess", root: Path | None = None, enabled: bool = True):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{name}"
        self.dir = (root or RUNS_DIR) / self.run_id
        self.enabled = enabled
        self.events: list[dict] = []
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.dir / "trace.jsonl", "a", encoding="utf-8")
        else:
            self._fh = None
        self._stage_start: dict[str, float] = {}

    def event(self, stage: str, **data: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "stage": stage, **data}
        self.events.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()
        if SETTINGS.verbose:
            short = {k: v for k, v in data.items() if k not in ("text",)}
            print(f"  [trace] {stage}: {json.dumps(short, default=str, ensure_ascii=False)[:160]}")

    def start(self, stage: str) -> None:
        self._stage_start[stage] = time.time()

    def end(self, stage: str, **data: Any) -> None:
        t0 = self._stage_start.pop(stage, None)
        self.event(stage, duration_ms=round((time.time() - t0) * 1000) if t0 else None, **data)

    def write_json(self, name: str, obj: Any) -> Path | None:
        if not self.enabled:
            return None
        p = self.dir / name
        p.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path | None:
        if not self.enabled:
            return None
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
