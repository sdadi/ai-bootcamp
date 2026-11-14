"""
matcher.py — the deterministic core: policy requirement × vendor fact → PASS / FAIL / UNKNOWN / NEEDS_REVIEW.

THIS FILE HAS NO LLM IN IT, ON PURPOSE.
The research layer may use a model to READ pages. The verdict is computed by code you can unit-test,
so that two runs on the same evidence always give the same verdict, and so that a reviewer can point at
the exact line of policy and the exact line of evidence behind every cell of the report.

COURSE MAP
  Deep Search course  Module 6  (from evidence to a clean structured result; provenance on every claim)
                      Module 10 (programmatic metrics: the verdicts here are what eval/run_eval.py scores)
  Production RAG      Module 9  ("Refusing": UNKNOWN beats a guess)

Rules are declared in data/requirements.yaml; each `rule:` name maps to one function below.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import yaml

from .config import REQUIREMENTS_FILE
from .policy_rag import PolicyLibrary
from .schemas import (AssessmentRequest, Overall, Requirement, RequirementResult, Verdict, VendorFact, VendorProfile)

TIER_RANK = {"T1": 1, "T2": 2, "T3": 3}

# Which questionnaire item (data/policies/security_questionnaire_template.md) to send when a rule is not PASS.
QUESTION_FOR_CLAUSE = {
    "SEC-4.2": "Q-SEC-4.2", "DP-3.1": "Q-DP-3.1", "DP-3.2": "Q-DP-3.1", "AI-2.1": "Q-AI-2.1", "AI-3.4": "Q-AI-3.4",
    "DP-5.4": "Q-DP-5.4", "SEC-7.1": "Q-SEC-7.1", "SEC-9.3": "Q-SEC-9.3", "DP-2.2": "Q-DP-2.2", "DP-4.1": "Q-DP-4.1",
}


def load_requirements(path: Path = REQUIREMENTS_FILE) -> list[Requirement]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Requirement.model_validate(r) for r in data["requirements"]]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("/", " ").replace("-", " ").split())


def _months_between(a: date, b: date) -> float:
    return abs((a - b).days) / 30.44


# ---------------------------------------------------------------------------
# Rule functions. Signature: (req, fact, ctx) -> (verdict, reason)
# ctx carries as_of, request, library — only approved_for_tier uses the library.
# ---------------------------------------------------------------------------

def rule_certification_valid(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not fact.value:
        return "UNKNOWN", "No security certification could be established from public sources."
    accepted = {_norm(a) for a in req.params.get("accepted", [])}
    max_age = int(req.params.get("max_age_months", 15))
    require_documented = bool(req.params.get("require_documented", True))
    as_of: date = ctx["as_of"]

    certs = list(fact.value)
    matches = [c for c in certs if _norm(c.get("name", "")) in accepted]
    generic_soc2 = [c for c in certs if _norm(c.get("name", "")) == "soc 2" and c.get("status") == "certified"]
    certified = [c for c in matches if c.get("status") == "certified"]

    def is_current(c: dict) -> bool:
        vu = c.get("valid_until")
        if vu:
            return date.fromisoformat(str(vu)) >= as_of
        d = c.get("date")
        return bool(d) and _months_between(as_of, date.fromisoformat(str(d))) <= max_age

    current = [c for c in certified if is_current(c)]
    if current:
        names = "; ".join(f"{c['name']} ({c.get('date') or 'date n/a'}{', valid to ' + str(c['valid_until']) if c.get('valid_until') else ''})" for c in current)
        if require_documented and fact.evidence_level != "documented":
            return "NEEDS_REVIEW", f"Vendor states {names}, but only as a claim. Request the report or certificate before relying on it."
        return "PASS", f"Current, independently issued: {names}."
    if certified:
        names = "; ".join(f"{c['name']} ({c.get('date') or 'undated'})" for c in certified)
        return "NEEDS_REVIEW", f"{names} is older than {max_age} months or undated. Request the current report."
    if generic_soc2:
        return "NEEDS_REVIEW", "Vendor says 'SOC 2' without stating Type I or Type II, a date or an auditor. A statement is not a report — request it."
    if matches:
        names = "; ".join(f"{c['name']} ({c.get('status')})" for c in matches)
        return "FAIL", f"Only {names}. A certification that is in progress or planned does not satisfy the clause."
    names = "; ".join(str(c.get("name")) for c in certs) or "none"
    return "FAIL", f"No accepted certification found (saw: {names}). Accepted: {', '.join(req.params.get('accepted', []))}."


def rule_region_includes(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not isinstance(fact.value, dict):
        return "UNKNOWN", "Storage regions could not be established from public sources."
    wanted = {_norm(x) for x in req.params.get("any_of", [])}
    available = {_norm(x) for x in fact.value.get("available", [])}
    planned = {_norm(x) for x in fact.value.get("planned", [])}
    if wanted & available:
        return "PASS", f"Customer content can be stored in {', '.join(sorted(wanted & available)).title()}. Regions: {fact.display}."
    if wanted & planned and not req.params.get("allow_planned", False):
        return "FAIL", f"{', '.join(sorted(wanted & planned)).title()} is only planned, which does not count. Available today: {', '.join(sorted(available)).title() or 'none stated'}."
    if available:
        return "FAIL", f"Storage regions are {', '.join(sorted(available)).title()}; none of the required regions ({', '.join(req.params.get('any_of', []))})."
    return "UNKNOWN", "The vendor does not state where content is stored."


def rule_value_known(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None:
        return "UNKNOWN", "Not stated by the vendor; record it in the review file."
    if isinstance(fact.value, dict) and not fact.value.get("available"):
        return "UNKNOWN", "Not stated by the vendor; record it in the review file."
    return "PASS", f"Recorded: {fact.display}."


def rule_enum_verdict(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or fact.value in (None, ""):
        return "UNKNOWN", "Not stated in the vendor's public material."
    v = str(fact.value)
    p = req.params
    if v in p.get("pass", []):
        return "PASS", fact.display
    if v in p.get("review", []):
        return "NEEDS_REVIEW", f"{fact.display}. Acceptable only with the conditions in {req.clause}; record how they are met."
    if v in p.get("fail", []):
        return "FAIL", fact.display
    return "UNKNOWN", f"Unrecognised value {v!r}; check the extraction."


def rule_list_disclosed(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not fact.value:
        return "UNKNOWN", "No sub-processor list could be found. Ask the vendor for it."
    flag_cats = set(req.params.get("review_if_category", []))
    flagged = [i for i in fact.value if i.get("category") in flag_cats]
    if flagged:
        names = ", ".join(f"{i['name']} ({i.get('country', '?')})" for i in flagged)
        return "NEEDS_REVIEW", f"{len(fact.value)} sub-processors disclosed. Model provider(s) present: {names} — confirm AI-2.1 applies to them too."
    return "PASS", f"{len(fact.value)} sub-processors disclosed; no third-party model provider."


def rule_retention_within(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or not isinstance(fact.value, dict):
        return "UNKNOWN", "Retention and deletion terms not stated."
    max_days = int(req.params.get("max_days", 30))
    default_days = fact.value.get("default_days")
    on_request = fact.value.get("deletion_on_request_days")
    if default_days is not None and int(default_days) <= max_days:
        return "PASS", f"Default retention {default_days} days (limit {max_days}). {fact.display}."
    if default_days is None and on_request is not None and int(on_request) <= max_days:
        return "NEEDS_REVIEW", f"Default retention not stated; deletion on request within {on_request} days. Record the commitment in the contract."
    if default_days is not None and on_request is not None and int(on_request) <= max_days:
        return "NEEDS_REVIEW", f"Default retention {default_days} days exceeds {max_days}, but deletion on request within {on_request} days is offered. Record the commitment in the contract."
    if default_days is not None and on_request is None:
        return "NEEDS_REVIEW", f"Default retention {default_days} days exceeds {max_days} and no deletion-on-request commitment was found. Ask the vendor."
    return "FAIL", f"Default retention {default_days} days and deletion on request {on_request} days both exceed {max_days}."


def rule_no_unresolved_incident(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    if fact is None or fact.value is None:
        return "UNKNOWN", "Incident history could not be established. Ask the vendor (and keep rumours out, SEC-9.4)."
    as_of: date = ctx["as_of"]
    lookback = as_of - timedelta(days=int(req.params.get("lookback_months", 24)) * 30)
    majors = {s.lower() for s in req.params.get("major_severities", ["major", "critical"])}
    recent = [i for i in fact.value if i.get("date") and date.fromisoformat(str(i["date"])) >= lookback]
    major = [i for i in recent if str(i.get("severity", "")).lower() in majors]
    without = [i for i in major if not i.get("postmortem_published")]
    if without:
        s = "; ".join(f"{i['date']}: {i['summary']}" for i in without)
        return "FAIL", f"Major incident without a published post-mortem: {s}."
    if major:
        s = "; ".join(f"{i['date']}: {i['summary']}" for i in major)
        return "NEEDS_REVIEW", f"Major incident(s) with a published post-mortem: {s}. Information Security must read the post-mortem before approval."
    minor = len(recent) - len(major)
    return "PASS", f"No major security incident in the look-back window." + (f" {minor} minor/availability incident(s) noted." if minor else "")


def rule_approved_for_tier(req: Requirement, fact: Optional[VendorFact], ctx: dict) -> tuple[Verdict, str]:
    library: PolicyLibrary = ctx["library"]
    request: AssessmentRequest = ctx["request"]
    entry = library.approved_vendor_entry(request.vendor, request.user_groups)
    if entry == "NO_ACCESS":
        return "UNKNOWN", "You do not have access to the Approved Vendor List. Ask Procurement for this vendor's approval status at this tier."
    if entry is None:
        return "NEEDS_REVIEW", "Not on the Approved Vendor List. This assessment is the input to that review; Procurement records the outcome (PROC-3.1)."
    if entry["status"] == "rejected":
        return "FAIL", f"Vendor was rejected on {entry.get('date')}: {entry.get('notes')}"
    if entry["status"] == "in_progress":
        return "NEEDS_REVIEW", f"A review is already in progress ({entry.get('notes')}). Do not start a second one; attach this assessment to it."
    approved = entry.get("tier", "")
    if TIER_RANK.get(approved, 0) >= TIER_RANK.get(request.tier, 3):
        return "PASS", f"Approved for {approved} on {entry.get('approved_on')} (plan: {entry.get('plan')}). {entry.get('notes')}"
    return "NEEDS_REVIEW", f"Approved for {approved} only (on {entry.get('approved_on')}); this request is for {request.tier}. Approval must be extended. {entry.get('notes')}"


RULES: dict[str, Callable[[Requirement, Optional[VendorFact], dict], tuple[Verdict, str]]] = {
    "certification_valid": rule_certification_valid,
    "region_includes": rule_region_includes,
    "value_known": rule_value_known,
    "enum_verdict": rule_enum_verdict,
    "list_disclosed": rule_list_disclosed,
    "retention_within": rule_retention_within,
    "no_unresolved_incident": rule_no_unresolved_incident,
    "approved_for_tier": rule_approved_for_tier,
}


class Matcher:
    def __init__(self, library: PolicyLibrary, requirements: list[Requirement] | None = None):
        self.library = library
        self.requirements = requirements or load_requirements()
        unknown = {r.rule for r in self.requirements} - set(RULES)
        if unknown:
            raise ValueError(f"requirements.yaml uses rules with no implementation: {sorted(unknown)}")

    def evaluate(self, profile: VendorProfile, request: AssessmentRequest) -> list[RequirementResult]:
        ctx = {"as_of": profile.as_of, "request": request, "library": self.library}
        results: list[RequirementResult] = []
        for req in self.requirements:
            fact = profile.get(req.field)
            if not req.applies(request.tier, request.users):
                verdict, reason = "NOT_APPLICABLE", f"Applies to tiers {', '.join(req.applies_when.tiers)}" + (
                    f" or {req.applies_when.or_min_users}+ users" if req.applies_when.or_min_users else "") + f"; this request is {request.tier}, {request.users} users."
            else:
                verdict, reason = RULES[req.rule](req, fact, ctx)
            q_id = None
            if verdict in ("UNKNOWN", "NEEDS_REVIEW"):
                q_id = QUESTION_FOR_CLAUSE.get(req.clause)
            results.append(RequirementResult(
                requirement_id=req.id, title=req.title, clause=req.clause, policy_doc=req.policy_doc,
                severity=req.severity, verdict=verdict, reason=reason,
                evidence_urls=list(fact.sources) if fact else [],
                evidence_as_of=fact.as_of if fact else None,
                evidence_level=fact.evidence_level if fact else None,
                vendor_question_id=q_id))
        return results

    @staticmethod
    def overall(results: list[RequirementResult]) -> tuple[Overall, str]:
        blockers = [r for r in results if r.verdict == "FAIL" and r.severity == "blocker"]
        if blockers:
            return "BLOCKED", "Blocking requirement(s) failed: " + ", ".join(f"{r.requirement_id} ({r.title})" for r in blockers) + "."
        majors = [r for r in results if r.verdict == "FAIL" and r.severity == "major"]
        open_items = [r for r in results if r.verdict in ("UNKNOWN", "NEEDS_REVIEW")]
        if majors or open_items:
            parts = []
            if majors:
                parts.append("failed major requirement(s) need an approver exception: " + ", ".join(r.requirement_id for r in majors))
            if open_items:
                parts.append(f"{len(open_items)} item(s) need evidence or a human decision: " + ", ".join(r.requirement_id for r in open_items))
            return "CONDITIONAL", "; ".join(parts).capitalize() + "."
        return "READY_FOR_APPROVAL", "Every applicable requirement passed on documented evidence. A human approver still decides (AI-5.1)."
