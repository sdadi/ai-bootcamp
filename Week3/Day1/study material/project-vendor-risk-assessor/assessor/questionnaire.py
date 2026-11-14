"""
questionnaire.py — turn every UNKNOWN / NEEDS_REVIEW into a question for the vendor.

This is the part reviewers love: instead of "we need more information", the report ends with the exact
questions to paste into an email, each tied to the policy clause it will settle. The question texts come
from the policy library (template NM-SEC-Q), so Information Security can edit them without touching code.

COURSE MAP  Production RAG Module 9 (structured output) — the questionnaire is a structured artefact
built from retrieved policy text, not free-form generation.
"""

from __future__ import annotations

from .policy_rag import PolicyLibrary
from .schemas import RequirementResult


def build_questions(results: list[RequirementResult], library: PolicyLibrary) -> list[dict]:
    template = {c.clause_id: c for c in library.chunks if c.doc_id == "NM-SEC-Q" and c.clause_id}
    out, seen = [], set()
    for r in results:
        if r.verdict not in ("UNKNOWN", "NEEDS_REVIEW") or not r.vendor_question_id:
            continue
        if r.vendor_question_id in seen:
            continue
        seen.add(r.vendor_question_id)
        chunk = template.get(r.vendor_question_id)
        text = chunk.text.strip() if chunk else f"Please provide evidence for requirement {r.clause} ({r.title})."
        out.append({"id": r.vendor_question_id, "requirement_id": r.requirement_id, "clause": r.clause,
                    "why": r.reason, "question": text})
    return out


def as_email(vendor: str, questions: list[dict], requester: str = "Nimbu Mart Information Security") -> str:
    if not questions:
        return ""
    lines = [f"Subject: Security review questions — {vendor}", "",
             f"Hello {vendor} team,", "",
             "We are reviewing your product for use at Nimbu Mart. Could you answer the following? "
             "Where a document exists (report, certificate, DPA), a link or attachment is ideal.", ""]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q['question']}")
        lines.append("")
    lines += ["Thank you,", requester]
    return "\n".join(lines)
