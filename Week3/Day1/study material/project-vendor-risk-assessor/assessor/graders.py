"""
graders.py — the two checks before an answer leaves the system.

COURSE MAP
  Agentic RAG guide  "agents.py → Answer Grader" (is the retrieved text relevant to the question?) and
                     "Hallucination Grader" (is every statement in the answer supported by the sources?).
  Production RAG     Module 9 "Reasoning, self-checks and guardrails"; Module 10 "Answer metrics" (faithfulness).

WHAT IS COMPLETE: offline heuristics (term overlap; sentence-level support) so the pipeline runs end to end.
LEARNER TODO — Milestone 3: port the two grader agents' prompts, call them through `LLM.structured()` with
  a {"verdict": "yes"|"no", "reason": ...} schema, and wire a retry: relevant=no → broaden the policy search
  (include more chunks); grounded=no → regenerate with a stricter prompt, then abstain.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from .llm import LLM
from .policy_rag import tokenize
from .schemas import PolicyAnswer


class GradeResult(BaseModel):
    verdict: bool
    reason: str = ""


RELEVANCE_SYSTEM = """You are the Answer Grader. Given a question and retrieved policy text, answer whether the text
contains information that answers the question. Be strict: related topic is not enough. Return {"verdict": true|false, "reason": "..."}."""

GROUNDED_SYSTEM = """You are the Hallucination Grader. Given sources and a draft answer, check that EVERY factual statement
in the answer is supported by the sources. If any statement is not, verdict is false. Return {"verdict": true|false, "reason": "..."}."""


def grade_relevance(question: str, context: str, llm: Optional[LLM] = None) -> GradeResult:
    def mock() -> dict:
        q, c = set(tokenize(question)), set(tokenize(context))
        overlap = len(q & c)
        return {"verdict": overlap >= max(1, len(q) // 3), "reason": f"{overlap}/{len(q)} question terms found in context (heuristic)"}

    return (llm or LLM()).structured(RELEVANCE_SYSTEM, f"Question: {question}\n\nContext:\n{context[:6000]}", GradeResult, stage="grade_relevance", mock=mock)


def grade_grounded(answer: str, sources: str, llm: Optional[LLM] = None) -> GradeResult:
    def mock() -> dict:
        # Every sentence of the answer must share most of its content words with the sources.
        src = set(tokenize(sources))
        body = re.sub(r"^[A-Z]{1,5}-[\d.]+:\s*", "", answer)
        body = re.sub(r"\(source:.*?\)\s*$", "", body).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\s*\|\s*", body) if len(tokenize(s)) >= 3]
        unsupported = [s for s in sentences if len(set(tokenize(s)) & src) / max(1, len(set(tokenize(s)))) < 0.6]
        return {"verdict": not unsupported, "reason": ("all sentences supported" if not unsupported else f"unsupported: {unsupported[0][:80]}") + " (heuristic)"}

    return (llm or LLM()).structured(GROUNDED_SYSTEM, f"Sources:\n{sources[:8000]}\n\nAnswer:\n{answer}", GradeResult, stage="grade_grounded", mock=mock)


def grade_policy_answer(pa: PolicyAnswer, llm: Optional[LLM] = None) -> PolicyAnswer:
    if pa.abstained or not pa.citations:
        return pa
    ctx = "\n\n".join(h.text for h in pa.citations)
    pa.grader_relevant = grade_relevance(pa.question, ctx, llm).verdict
    pa.grader_grounded = grade_grounded(pa.answer, ctx, llm).verdict
    if not pa.grader_grounded:
        pa.abstained = True
        pa.answer = "I found related policy text but could not produce an answer that is fully supported by it. Please read the cited clauses directly."
    return pa
