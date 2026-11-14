"""
tracing.py — write down everything the agent does.

You cannot debug or evaluate an agent you cannot see. Every important step
(LLM call, tool call, validation decision, merge decision) is appended as one
JSON line to runs/<run-id>/trace.jsonl. Open that file after a run — it is
the best way to understand what happened and why.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR

_lock = threading.Lock()
_state: dict = {"path": None, "events": []}


def start_run(name: str = "run") -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + f"-{name}"
    folder = RUNS_DIR / run_id
    folder.mkdir(parents=True, exist_ok=True)
    _state["path"] = folder / "trace.jsonl"
    _state["events"] = []
    return folder


def trace(event: str, **data) -> None:
    record = {"t": datetime.now().isoformat(timespec="milliseconds"), "event": event, **data}
    with _lock:
        _state["events"].append(record)
        if _state["path"] is not None:
            with open(_state["path"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")


def events(kind: str | None = None) -> list[dict]:
    return [e for e in _state["events"] if kind is None or e["event"] == kind]
