"""
schemas.py — the "contracts" between agents.

Every object that moves from one stage to the next is a Pydantic model.
If an agent (or an LLM) produces something that does not fit the contract,
Pydantic raises a ValidationError *before* the bad data spreads downstream.

Read this file top to bottom once: it is the vocabulary of the whole system.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

# The facts our company-research agent tries to establish.
FieldName = Literal[
    "ceo",
    "founded_year",
    "headquarters",
    "latest_funding",
    "regulatory_status",
    "pricing_domestic_cards",
    "uptime_90d",
    "major_outage",
    "security_incident",
    "merchant_rating",
]

FIELD_DESCRIPTIONS: dict[str, str] = {
    "ceo": "Who runs the company now (and who ran it before).",
    "founded_year": "Year the company was founded.",
    "headquarters": "City of the headquarters.",
    "latest_funding": "Most recent funding round and amount, e.g. 'Series B — $40M'.",
    "regulatory_status": "Licence / authorisation status with the payments regulator.",
    "pricing_domestic_cards": "Fee per transaction on domestic cards, e.g. '2.0%'.",
    "uptime_90d": "Measured platform uptime over the last 90 days.",
    "major_outage": "Date of the most recent major outage.",
    "security_incident": "Most recent disclosed security incident and its impact.",
    "merchant_rating": "Average merchant review rating, e.g. '3.9/5'.",
}

SourceType = Literal[
    "regulator",
    "official",
    "news",
    "press_release",
    "news_syndicated",
    "video_transcript",
    "review_site",
    "forum",
    "blog",
    "unknown",
]


# ---------------------------------------------------------------------------
# 1. Planning layer
# ---------------------------------------------------------------------------
class EntityAnchor(BaseModel):
    """What makes OUR target different from look-alikes (same-name companies)."""

    name: str = Field(description="Name as the user wrote it, e.g. 'NimbusPay'")
    country: str = Field(default="", description="e.g. 'India'")
    city: str = Field(default="", description="e.g. 'Bengaluru'")
    industry_keywords: list[str] = Field(default_factory=list, description="e.g. ['payment gateway']")
    official_domain: str = Field(default="", description="e.g. 'nimbuspay.example'")
    exclude: list[str] = Field(default_factory=list, description="Known look-alikes to ignore")


class SubQuestion(BaseModel):
    id: str = Field(description="Short id like 'q1'")
    question: str
    fields: list[FieldName] = Field(description="Which report fields this sub-question fills")
    lens: Literal["entity", "claim", "time", "source_type", "decision"] = "claim"
    priority: Literal["high", "medium", "low"] = "medium"
    freshness_days: int = Field(default=365, description="Evidence older than this is 'stale' for this question")
    preferred_sources: list[SourceType] = Field(default_factory=list)
    high_stakes: bool = Field(default=False, description="Needs 2 independent sources or 1 regulator/official source")


class ResearchPlan(BaseModel):
    goal: str
    entity: EntityAnchor
    sub_questions: list[SubQuestion] = Field(min_length=1)
    source_strategy: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    stop_criteria: list[str] = Field(default_factory=list)


class PlanReview(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    fixes_applied: list[str] = Field(default_factory=list)


class PlanDecision(BaseModel):
    """Output of the Plan Evaluator after each iteration."""

    action: Literal["continue", "stop"]
    reasons: list[str]
    open_sub_questions: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Executor and tool layer
# ---------------------------------------------------------------------------
class SearchQuery(BaseModel):
    """Input contract for the search tool (the LLM must produce exactly this)."""

    query: str = Field(min_length=3, max_length=300)
    max_results: int = Field(default=5, ge=1, le=20)
    date_from: Optional[date] = None


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    rank: int
    query: str
    published_date: Optional[date] = None
    retrieved_at: datetime


class SourceMeta(BaseModel):
    """Provenance: where a claim came from and WHEN (four different dates!)."""

    doc_id: str
    url: str
    domain: str
    title: str
    source_type: SourceType
    publisher: str = ""
    publication_date: Optional[date] = None
    last_updated: Optional[date] = None
    retrieval_date: date


class Document(BaseModel):
    meta: SourceMeta
    text: str
    flags: list[str] = Field(default_factory=list, description="e.g. 'prompt_injection_suspected'")


class ExtractedClaim(BaseModel):
    """What the extraction step (LLM or rules) pulls out of one document."""

    field: FieldName
    value: str = Field(min_length=1)
    evidence_snippet: str = Field(min_length=5, description="Exact words from the source")
    event_date: Optional[date] = Field(default=None, description="When the fact happened, if the text says")


class ClaimList(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class Evidence(BaseModel):
    """A claim + its provenance + the validator's scores. The unit everything works on."""

    id: str
    sub_question_id: str
    claim: ExtractedClaim
    source: SourceMeta
    identity_match: float = 0.0
    reliability: float = 0.0
    freshness: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    origin_id: str = Field(default="", description="Original source id if this page is a copy")
    status: Literal["current", "historical", "unknown"] = "unknown"
    flags: list[str] = Field(default_factory=list)
    accepted: bool = True


# ---------------------------------------------------------------------------
# 3. Merging and final answer
# ---------------------------------------------------------------------------
class MergedClaim(BaseModel):
    field: FieldName
    value: str
    as_of: Optional[date] = None
    confidence: float
    status: Literal["current", "historical", "unknown"]
    supporting_sources: list[str]
    independent_sources: int
    evidence_ids: list[str]


class ConflictRecord(BaseModel):
    field: FieldName
    candidates: list[dict]
    resolution: Literal["accepted_one", "rejected_weak", "time_separated", "unresolved"]
    chosen_value: Optional[str] = None
    rationale: str


class Finding(BaseModel):
    field: FieldName
    value: str
    confidence: Literal["high", "medium", "low"]
    status: Literal["current", "historical", "unknown"]
    as_of: Optional[date] = None
    sources: list[str]


class FinalReport(BaseModel):
    question: str
    as_of: date
    answer: str
    confidence: Literal["high", "medium", "low"]
    key_findings: list[Finding]
    historical_findings: list[Finding] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    freshness_notes: list[str] = Field(default_factory=list)
    security_notes: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    run_stats: dict = Field(default_factory=dict)


# A tiny example of HttpUrl validation used in Lab 2.
class FetchRequest(BaseModel):
    url: HttpUrl
    max_chars: int = Field(default=6000, ge=500, le=50000)
