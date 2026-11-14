"""
memory.py — long-term memory that makes the NEXT run better.

Short-term context (the current plan, evidence, open gaps) lives in Python
objects during a run. Long-term memory survives between runs in
memory/lessons.json and stores REUSABLE lessons, not raw page text:

  source_stats   per-domain: how often its claims were accepted vs rejected
  lessons        short heuristics the planner reads before planning
  look_alikes    entities that were confused with a target before

Guardrails (see the course): store dates and evidence counts with every lesson,
never let memory override fresh evidence, avoid storing personal data, and
re-validate time-sensitive facts instead of trusting memory.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "memory" / "lessons.json"


class Memory:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.data = {"source_stats": {}, "lessons": [], "look_alikes": {}}
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))

    # ---- reading ---------------------------------------------------------
    def planner_tips(self, limit: int = 6) -> list[str]:
        lessons = sorted(self.data["lessons"], key=lambda l: -l.get("times_seen", 1))
        return [l["text"] for l in lessons[:limit]]

    def known_look_alikes(self, entity_name: str) -> list[str]:
        return self.data["look_alikes"].get(entity_name.lower(), [])

    def blocked_domains(self) -> set[str]:
        """Domains that earlier runs caught serving a different entity or a prompt injection."""
        bad = {"identity_mismatch", "prompt_injection"}
        return {d for d, s in self.data["source_stats"].items()
                if s["rejected"] and not s["accepted"] and bad & set(",".join(s["reasons"]).split(","))}

    def domain_track_record(self, domain: str) -> dict | None:
        return self.data["source_stats"].get(domain)

    # ---- writing ---------------------------------------------------------
    def record_source(self, domain: str, accepted: bool, reason: str = "") -> None:
        s = self.data["source_stats"].setdefault(domain, {"accepted": 0, "rejected": 0, "reasons": []})
        s["accepted" if accepted else "rejected"] += 1
        if reason and reason not in s["reasons"]:
            s["reasons"].append(reason)

    def add_lesson(self, text: str, evidence: str) -> None:
        for lesson in self.data["lessons"]:
            if lesson["text"] == text:
                lesson["times_seen"] = lesson.get("times_seen", 1) + 1
                lesson["last_seen"] = date.today().isoformat()
                return
        self.data["lessons"].append({"text": text, "evidence": evidence, "times_seen": 1,
                                     "first_seen": date.today().isoformat(), "last_seen": date.today().isoformat()})

    def add_look_alike(self, entity_name: str, look_alike: str) -> None:
        items = self.data["look_alikes"].setdefault(entity_name.lower(), [])
        if look_alike not in items:
            items.append(look_alike)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.data = {"source_stats": {}, "lessons": [], "look_alikes": {}}
        if self.path.exists():
            self.path.unlink()
