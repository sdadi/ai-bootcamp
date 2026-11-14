"""
schemas.py — the contracts between every stage of the assessor.

Everything that moves between stages is a Pydantic model, exactly as in the Deep Search course
(deepsearch/schemas.py, Module 4 "Schema enforcement with Pydantic"). If an LLM or a tool produces
something that does not fit, validation fails HERE, before bad data spreads.

Read top to bottom once; it is the vocabulary of the whole project.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Vendor facts — what the research layer tries to establish about a vendor.
#    (Deep Search course Module 12 "Adapt it to your own domain": this is place 1 of 4.)
# ---------------------------------------------------------------------------

VendorField = Literal[
    "legal_entity",               # identity anchor: which company are we actually looking at?
    "security_certifications",    # SOC 2 Type II / ISO 27001, with dates and evidence level
    "data_regions",               # where customer content is stored (available vs planned)
    "subprocessors",              # who else touches the data, incl. model providers
    "trains_on_customer_data",    # no | no_on_paid_plans | opt_out_available | yes
    "data_retention",             # default retention + deletion on request (days)
    "sso_saml",                   # all_plans | paid_plans | enterprise_only | none
    "security_incidents",         # last 24 months, with post-mortem status
    "dpa_available",              # yes | on_request | no
]

VENDOR_FIELDS: tuple[str, ...] = VendorField.__args__  # type: ignore[attr-defined]

FIELD_DESCRIPTIONS: dict[str, str] = {
    "legal_entity": "The legal entity, country and HQ city of the company that runs the product.",
    "security_certifications": "Independent security attestations: SOC 2 (Type I/II) report dates, ISO 27001 certificate, issuer, validity.",
    "data_regions": "Countries/regions where customer content is stored at rest; separate 'available now' from 'planned'.",
    "subprocessors": "Third parties that process customer content for the vendor: name, purpose, category, country.",
    "trains_on_customer_data": "Whether customer content is used to train or improve models, and on which plans.",
    "data_retention": "Default retention after contract end and whether deletion on request is offered, in days.",
    "sso_saml": "Whether SAML/OIDC single sign-on is offered and on which plan.",
    "security_incidents": "Security incidents affecting customer data in the last 24 months and whether a post-mortem was published.",
    "dpa_available": "Whether a Data Processing Agreement is published, available on request, or not offered.",
}

# Fields where a wrong answer is expensive: we demand documented evidence from a non-forum source.
HIGH_STAKES_FIELDS: frozenset[str] = frozenset({"security_certifications", "security_incidents", "trains_on_customer_data", "data_regions"})

# Fields whose value is a LIST that should be merged (union) across sources rather than "best source wins".
LIST_FIELDS: frozenset[str] = frozenset({"security_incidents"})

EvidenceLevel = Literal[
    "documented",   # a report, certificate, registry entry, or the vendor's own policy/legal page
    "claimed",      # marketing text with no verifiable artefact ("we are SOC 2 compliant")
    "third_party",  # a named news outlet or review site reporting it
    "rumour",       # forum, social media, unnamed sources
]

SourceType = Literal[
    "regulator",        # certification body registry, government register
    "official",         # the vendor's own site: trust centre, legal, status page
    "news",             # named news outlet
    "news_syndicated",  # a copy of someone else's content (press release reprint, directory listing)
    "review_site",
    "forum",
    "blog",
    "unknown",
]


class SourceMeta(BaseModel):
    id: str
    url: str
    title: str
    source_type: SourceType = "unknown"
    publisher: str = ""
    published_date: Optional[date] = None
    last_updated: Optional[date] = None

    @property
    def effective_date(self) -> Optional[date]:
        return self.last_updated or self.published_date


class Claim(BaseModel):
    """One fact pulled from ONE page. This is what the extraction LLM returns (or `mock_claims` offline)."""

    field: VendorField
    value: Any
    display: str = Field(description="Short human-readable version of value")
    quote: str = Field(default="", description="The sentence(s) in the page that support the value")
    evidence_level: EvidenceLevel = "claimed"


class ClaimList(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class Evidence(BaseModel):
    """A claim after validation: who said it, how reliable, how fresh, final score."""

    source: SourceMeta
    claim: Claim
    authority: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    level_weight: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)
    duplicate_of: Optional[str] = Field(default=None, description="Source id this page copies, if any")


class VendorFact(BaseModel):
    field: VendorField
    value: Any
    display: str
    confidence: float = Field(ge=0, le=1)
    as_of: Optional[date] = None
    evidence_level: EvidenceLevel
    sources: list[str] = Field(default_factory=list, description="URLs that support the chosen value")
    conflicts: list[str] = Field(default_factory=list, description="Other values seen, with their sources")


class VendorProfile(BaseModel):
    vendor: str
    official_domain: Optional[str] = None
    as_of: date
    facts: dict[str, VendorFact] = Field(default_factory=dict)
    quarantined: list[str] = Field(default_factory=list, description="URLs removed for prompt injection")
    rejected: list[str] = Field(default_factory=list, description="URLs about a different company (identity gate)")
    dropped: list[str] = Field(default_factory=list, description="Claims dropped as rumour / unreliable, with reason")
    pages_read: int = 0
    notes: list[str] = Field(default_factory=list)

    def get(self, field: str) -> Optional[VendorFact]:
        return self.facts.get(field)


# ---------------------------------------------------------------------------
# 2. Policy library — what the RAG layer returns (Production RAG course Modules 8, 9, 11).
# ---------------------------------------------------------------------------


class PolicyHit(BaseModel):
    doc: str
    title: str
    doc_id: str
    clause_id: str = ""
    heading: str = ""
    text: str
    score: float
    version: str = ""
    status: str = "active"
    trust: str = "trusted"
    access: list[str] = Field(default_factory=list)


class PolicyAnswer(BaseModel):
    question: str
    answer: str
    citations: list[PolicyHit] = Field(default_factory=list)
    abstained: bool = False
    mode: Literal["extractive", "llm"] = "extractive"
    grader_relevant: Optional[bool] = None
    grader_grounded: Optional[bool] = None


# ---------------------------------------------------------------------------
# 3. Requirements and verdicts — the deterministic core (matcher.py).
# ---------------------------------------------------------------------------

Tier = Literal["T1", "T2", "T3"]
TIER_NAMES = {"T1": "Public", "T2": "Internal", "T3": "Customer data"}

Verdict = Literal["PASS", "FAIL", "UNKNOWN", "NEEDS_REVIEW", "NOT_APPLICABLE"]
Severity = Literal["blocker", "major", "minor"]


class AppliesWhen(BaseModel):
    tiers: list[Tier] = Field(default_factory=lambda: ["T1", "T2", "T3"])
    or_min_users: Optional[int] = Field(default=None, description="Rule also applies when users >= this, whatever the tier")


class Requirement(BaseModel):
    id: str
    title: str
    policy_doc: str
    clause: str
    field: str
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)
    severity: Severity = "major"
    applies_when: AppliesWhen = Field(default_factory=AppliesWhen)

    def applies(self, tier: str, users: int) -> bool:
        if tier in self.applies_when.tiers:
            return True
        if self.applies_when.or_min_users is not None and users >= self.applies_when.or_min_users:
            return True
        return False


class RequirementResult(BaseModel):
    requirement_id: str
    title: str
    clause: str
    policy_doc: str
    severity: Severity
    verdict: Verdict
    reason: str
    evidence_urls: list[str] = Field(default_factory=list)
    evidence_as_of: Optional[date] = None
    evidence_level: Optional[EvidenceLevel] = None
    vendor_question_id: Optional[str] = Field(default=None, description="Questionnaire item to send if UNKNOWN / NEEDS_REVIEW")


# ---------------------------------------------------------------------------
# 4. The request, the route, and the final report (router.py, graph.py, report.py).
# ---------------------------------------------------------------------------

Route = Literal["policy", "vendor", "assessment", "refuse"]


class RouteDecision(BaseModel):
    route: Route
    vendor: Optional[str] = None
    reason: str = ""


class AssessmentRequest(BaseModel):
    question: str
    vendor: str
    official_domain: Optional[str] = None
    tier: Tier = "T3"
    users: int = 1
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"])
    requested_by: str = "unknown"


Overall = Literal["BLOCKED", "CONDITIONAL", "READY_FOR_APPROVAL"]


class Approval(BaseModel):
    status: Literal["pending", "approved", "rejected", "not_required"] = "pending"
    approver_roles: list[str] = Field(default_factory=list)
    decided_by: Optional[str] = None
    decided_on: Optional[date] = None
    comment: str = ""


class AssessmentReport(BaseModel):
    request: AssessmentRequest
    profile: VendorProfile
    results: list[RequirementResult]
    overall: Overall
    overall_reason: str
    open_questions: list[dict] = Field(default_factory=list)   # {id, question, requirement_id}
    approval: Approval = Field(default_factory=Approval)
    guardrail_notes: list[str] = Field(default_factory=list)
    run_id: str = ""
    cost_usd: float = 0.0
    llm_calls: int = 0
    duration_s: float = 0.0
