"""
Chapter 12 - Tracing, latency and cost.

Every request gets a trace id. Every stage (embed, search, rerank, generate) is a span with its
duration and useful attributes (how many chunks, scores, token counts). Traces are written as
JSON lines to logs/traces.jsonl. Tools like Langfuse, Arize Phoenix, LangSmith or W&B Weave do the
same thing with a UI; OpenTelemetry has (still evolving) "GenAI" conventions for these spans.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class Trace:
    def __init__(self, name: str, sink: Path | None, **attrs):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.attrs = dict(attrs)
        self.spans: list[dict] = []
        self._t0 = time.perf_counter()
        self._sink = sink

    @contextmanager
    def span(self, name: str, **attrs):
        t0 = time.perf_counter()
        record = {"name": name, **attrs}
        try:
            yield record  # the caller may add attributes: record["n_hits"] = 5
        finally:
            record["ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self.spans.append(record)

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)

    def end(self) -> dict:
        data = {"trace_id": self.id, "name": self.name, "ts": time.time(),
                "total_ms": round((time.perf_counter() - self._t0) * 1000, 2),
                "attrs": self.attrs, "spans": self.spans}
        if self._sink:
            self._sink.parent.mkdir(parents=True, exist_ok=True)
            with self._sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        return data


class Tracer:
    def __init__(self, sink: Path | None = None):
        self.sink = sink

    def start(self, name: str, **attrs) -> Trace:
        return Trace(name, self.sink, **attrs)


def percentiles(values: list[float], ps=(50, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    return {f"p{p}": round(float(np.percentile(values, p)), 1) for p in ps}


def estimate_cost(input_tokens: int, output_tokens: int, usd_per_m_in: float, usd_per_m_out: float) -> float:
    """Cost of one LLM call in USD. Prices are per 1 million tokens (check your provider's page)."""
    return input_tokens / 1e6 * usd_per_m_in + output_tokens / 1e6 * usd_per_m_out
