"""
budget.py — tokens, money, time and tool calls are a deep-search agent's fuel.

Multi-agent research systems burn far more tokens than a chat. Without a
budget, an agent that "keeps searching until it is confident" can loop for a
long time. The Budget object below is checked by the orchestrator before each
step, and every LLM call / tool call is charged to it.

Prices: the table is EXAMPLE pricing per 1 million tokens so the maths is
concrete. Real prices differ by provider and change often — look up your
provider's pricing page and edit PRICE_PER_MTOK or set env vars.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

# (input $/1M tokens, output $/1M tokens) — EXAMPLE numbers for teaching.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "mock-llm": (0.10, 0.50),  # priced like a small model so offline runs show realistic cost maths
    "small": (0.10, 0.50),
    "large": (3.00, 15.00),
}


def price_for(model: str) -> tuple[float, float]:
    env_in, env_out = os.getenv("PRICE_IN_PER_MTOK"), os.getenv("PRICE_OUT_PER_MTOK")
    if env_in and env_out:
        return float(env_in), float(env_out)
    m = model.lower()
    if m in PRICE_PER_MTOK:
        return PRICE_PER_MTOK[m]
    if any(k in m for k in ("mini", "flash", "haiku", "nano", "small", "8b", "20b", "lite")):
        return PRICE_PER_MTOK["small"]
    return PRICE_PER_MTOK["large"]


def estimate_tokens(text: str) -> int:
    """Rule of thumb for English: ~4 characters per token. Good enough for budgets."""
    return max(1, len(text) // 4)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_usd: float = 0.50
    max_tokens: int = 400_000
    max_seconds: float = 300.0
    max_tool_calls: int = 120
    # running totals
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0
    started: float = field(default_factory=time.time)
    by_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- charging -----------------------------------------------------------
    def charge_llm(self, model: str, input_tokens: int, output_tokens: int, stage: str = "other") -> None:
        p_in, p_out = price_for(model)
        cost = input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.usd += cost
            self.llm_calls += 1
            s = self.by_stage.setdefault(stage, {"llm_calls": 0, "tokens": 0, "usd": 0.0, "tool_calls": 0})
            s["llm_calls"] += 1
            s["tokens"] += input_tokens + output_tokens
            s["usd"] += cost

    def charge_tool(self, stage: str = "tools") -> None:
        with self._lock:
            self.tool_calls += 1
            s = self.by_stage.setdefault(stage, {"llm_calls": 0, "tokens": 0, "usd": 0.0, "tool_calls": 0})
            s["tool_calls"] += 1

    # -- checking -----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def remaining_fraction(self) -> float:
        """0.0 = nothing left, 1.0 = untouched. The tightest limit wins."""
        fractions = [
            1 - self.usd / self.max_usd if self.max_usd else 1,
            1 - self.total_tokens / self.max_tokens if self.max_tokens else 1,
            1 - self.elapsed / self.max_seconds if self.max_seconds else 1,
            1 - self.tool_calls / self.max_tool_calls if self.max_tool_calls else 1,
        ]
        return max(0.0, min(fractions))

    def exhausted(self) -> str | None:
        """Return a human-readable reason if any limit is hit, else None."""
        if self.usd >= self.max_usd:
            return f"cost limit reached (${self.usd:.4f} of ${self.max_usd:.2f})"
        if self.total_tokens >= self.max_tokens:
            return f"token limit reached ({self.total_tokens:,} of {self.max_tokens:,})"
        if self.elapsed >= self.max_seconds:
            return f"time limit reached ({self.elapsed:.0f}s of {self.max_seconds:.0f}s)"
        if self.tool_calls >= self.max_tool_calls:
            return f"tool-call limit reached ({self.tool_calls} of {self.max_tool_calls})"
        return None

    def summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_usd": round(self.usd, 5),
            "seconds": round(self.elapsed, 2),
            "by_stage": {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in self.by_stage.items()},
        }
