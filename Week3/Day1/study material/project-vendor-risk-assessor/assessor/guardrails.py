"""
guardrails.py — checks that run around the agent, not inside it.

Three places a check can run: on the INPUT (before any model sees it), on TOOL OUTPUT (web pages —
handled by vendor_research.scan_injection), and on the OUTPUT (before the user sees it).

COURSE MAP
  Production RAG  Module 11 (PII redaction; prompt injection through documents)
  Deep Search     Module 8  (defence in depth; the "lethal trifecta": private data + untrusted content + a way out)

WHAT IS COMPLETE: regex-based input redaction (secrets, personal identifiers), an "approve" interception, and
an output policy (never say "approved"; every PASS/FAIL must carry evidence).
LEARNER TODO — Milestone 4: add an LLM-based injection classifier as a second layer behind the regexes
(Deep Search Lab 9), and a red-team suite of 15 attack questions that must all be caught (put it in tests/).
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import AssessmentReport

SECRET_PATTERNS = {
    "api_key": re.compile(r"\b(sk-[A-Za-z0-9]{12,}|AKIA[0-9A-Z]{16}|gsk_[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9-]{10,})\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "aadhaar_like": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone_in": re.compile(r"\b(\+91[\s-]?)?[6-9]\d{9}\b"),
}
APPROVE_RE = re.compile(r"\b(approve|mark (it|this|that|the)( vendor| tool)? (as )?approved|sign (it )?off|bypass|skip the (checks|review))\b", re.I)


def redact_input(text: str) -> tuple[str, list[str]]:
    """Remove secrets and personal identifiers BEFORE the text reaches a model or a log (AI-4.1)."""
    notes = []
    for name, pat in SECRET_PATTERNS.items():
        if pat.search(text):
            text = pat.sub(f"[{name.upper()} REDACTED]", text)
            notes.append(f"input: {name} redacted")
    return text, notes


def check_intent(text: str) -> Optional[str]:
    """Return a refusal message if the user asks the assistant to do something only a human may do."""
    if APPROVE_RE.search(text):
        return ("This assistant does not approve vendors or skip checks. It produces an assessment for the approvers "
                "named in AI-5.1. Ask it to assess the tool instead.")
    return None


FORBIDDEN_OUTPUT = re.compile(r"\b(is approved|approved for use|fully compliant|safe to use|you can go ahead)\b", re.I)


def enforce_output_policy(report: AssessmentReport) -> AssessmentReport:
    """The output guardrail: wording and evidence rules the report must meet before anyone reads it."""
    notes = report.guardrail_notes
    if FORBIDDEN_OUTPUT.search(report.overall_reason):
        report.overall_reason = FORBIDDEN_OUTPUT.sub("[wording removed by output guardrail]", report.overall_reason)
        notes.append("output: approval-like wording removed from the summary")
    for r in report.results:
        # PROC-1.1 is decided from the internal Approved Vendor List, so it carries a policy citation, not a URL.
        if r.verdict in ("PASS", "FAIL") and not r.evidence_urls and r.requirement_id != "PROC-1.1":
            r.verdict = "UNKNOWN"
            r.reason = "Verdict withdrawn by output guardrail: no evidence URL attached. " + r.reason
            notes.append(f"output: {r.requirement_id} downgraded to UNKNOWN (no evidence)")
    if report.approval.status == "approved" and not report.approval.decided_by:
        report.approval.status = "pending"
        notes.append("output: approval without a named approver reset to pending")
    return report
