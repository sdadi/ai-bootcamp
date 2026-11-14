"""
router.py — decide which line a question takes: policy library, vendor research, full assessment, or refuse.

COURSE MAP
  Agentic RAG guide  "agents.py → The Router" and "tasks.py → Task 1: route". Same idea, one more route:
  the course routes between "our PDF" and "the web"; here we route between "our policies", "the vendor's
  web presence" and "both, then match" (an assessment).

WHAT IS COMPLETE: a rule-based router that works offline and is good enough for the demo questions.
LEARNER TODO — Milestone 3: replace `decide()`'s mock with an LLM router. Port the Router agent's
  backstory from the Agentic RAG kit (the "semantic meaning" sentence matters — the course's exercise
  "Break the Router on purpose" shows why), add the two new routes, and return `RouteDecision` via
  `LLM.structured()` so a malformed answer is repaired, not trusted.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .llm import LLM
from .schemas import RouteDecision

ROUTER_SYSTEM = """You are the Router for Nimbu Mart's vendor risk assistant. Classify the user's question:
- "policy": asks what OUR internal policy says (clauses, tiers, who approves, what is required). Needs no vendor facts.
- "vendor": asks a factual question about a specific vendor/tool (certifications, regions, incidents) with no policy judgement.
- "assessment": asks whether a team CAN/MAY/SHOULD use a tool, or to assess/review/check a vendor against policy. Needs both.
- "refuse": not about vendors, tools or policy, or asks you to approve something (only humans approve).
Judge by the semantic meaning of the question, not by keywords. Return {"route": ..., "vendor": "<name or null>", "reason": "..."}."""

ASSESS_RE = re.compile(r"\b(can|may|should|could|allowed|ok(ay)? to|safe to|assess|review|evaluate|check|onboard|approve)\b.*\b(use|using|adopt|buy|onboard|vendor|tool)\b|\b(assess|review|evaluate|check)\b", re.I)
POLICY_RE = re.compile(r"\b(policy|policies|clause|tier|t[123]\b|who approves|required|requirement|must|dpdp|our (rules|process))\b", re.I)


def guess_vendor(question: str, known_vendors: Iterable[str]) -> Optional[str]:
    q = question.lower()
    for v in known_vendors:
        if v.lower() in q:
            return v
    return None


def decide(question: str, known_vendors: Iterable[str], llm: Optional[LLM] = None) -> RouteDecision:
    vendor = guess_vendor(question, known_vendors)

    def mock() -> dict:
        if re.search(r"\b(approve|mark .* approved|sign off)\b", question, re.I) and not ASSESS_RE.search(question):
            return {"route": "refuse", "vendor": vendor, "reason": "Only a human approver can approve a vendor (AI-5.1)."}
        if vendor and ASSESS_RE.search(question):
            return {"route": "assessment", "vendor": vendor, "reason": "Asks whether a named tool may be used → needs policy + vendor facts."}
        if vendor and POLICY_RE.search(question):
            return {"route": "policy", "vendor": vendor, "reason": "Asks what OUR records/policy say about a named vendor (e.g. approval tier) → policy library."}
        if vendor:
            return {"route": "vendor", "vendor": vendor, "reason": "Factual question about a named vendor."}
        return {"route": "policy", "vendor": None, "reason": "No vendor named → internal policy question."}

    llm = llm or LLM()
    return llm.structured(ROUTER_SYSTEM, f"Known vendors: {', '.join(known_vendors)}\nQuestion: {question}", RouteDecision,
                          stage="route", mock=mock)
