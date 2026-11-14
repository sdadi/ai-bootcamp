"""
validator.py — evidence evaluation BEFORE evidence becomes knowledge.

For every Evidence object we compute four numbers between 0 and 1:

  authority     how much this TYPE of source should be trusted for THIS field
  freshness     how recent the claim is, decaying with a field-specific half-life
  identity      is this page about OUR company (not a same-name look-alike)?
  completeness  does the evidence carry a date, a real snippet, a known publisher?

  reliability = 0.45*authority + 0.25*freshness + 0.15*identity + 0.15*completeness

We also detect COPIES (syndicated press releases): five sites repeating one
press release are ONE origin, not five confirmations.

Claim-level confidence (how sure we are that a VALUE is right) is computed
later in merger.py, because it depends on how many independent origins agree.
"""

from __future__ import annotations

import re
from datetime import date

from .schemas import Document, EntityAnchor, Evidence

AUTHORITY: dict[str, float] = {
    "regulator": 1.00,
    "official": 0.85,
    "news": 0.75,
    "video_transcript": 0.60,
    "press_release": 0.55,
    "review_site": 0.50,
    "news_syndicated": 0.45,
    "forum": 0.35,
    "blog": 0.25,
    "unknown": 0.20,
}
# Some source types are the best authority for a particular field.
FIELD_AUTHORITY_OVERRIDES: dict[tuple[str, str], float] = {
    ("merchant_rating", "review_site"): 0.80,  # the review site IS the source of truth for ratings
    ("security_incident", "news"): 0.80,       # independent reporting matters for incidents
    ("major_outage", "forum"): 0.50,           # users noticing an outage in real time is useful
}
# Half-life in days: after this many days a claim is worth half as "fresh".
HALF_LIFE_DAYS: dict[str, int] = {
    "ceo": 365,
    "founded_year": 36500,      # basically never changes
    "headquarters": 1825,
    "latest_funding": 365,
    "regulatory_status": 365,
    "pricing_domestic_cards": 180,
    "uptime_90d": 90,
    "major_outage": 365,
    "security_incident": 730,
    "merchant_rating": 180,
}
TIME_SENSITIVE_FIELDS = {"ceo", "latest_funding", "regulatory_status", "pricing_domestic_cards",
                         "uptime_90d", "major_outage", "security_incident", "merchant_rating"}
HIGH_STAKES_FIELDS = {"regulatory_status", "security_incident"}
WEIGHTS = {"authority": 0.45, "freshness": 0.25, "identity": 0.15, "completeness": 0.15}


def claim_date(ev: Evidence) -> date | None:
    """The date a claim is ABOUT: event date if known, else last update, else publication."""
    return ev.claim.event_date or ev.source.last_updated or ev.source.publication_date


def authority_score(field: str, source_type: str) -> float:
    return FIELD_AUTHORITY_OVERRIDES.get((field, source_type), AUTHORITY.get(source_type, 0.2))


def freshness_score(field: str, when: date | None, as_of: date) -> float:
    if when is None:
        return 0.3  # undated evidence is a risk: we cannot tell if it is current
    age = max(0, (as_of - when).days)
    return round(0.5 ** (age / HALF_LIFE_DAYS.get(field, 365)), 3)


def identity_score(doc: Document, entity: EntityAnchor) -> float:
    """Share of identity anchors that the page matches. 0.0 if it names a known look-alike."""
    text = (doc.meta.title + " " + doc.text).lower()
    if any(x and (x.lower() in text or x.lower() == doc.meta.domain) for x in entity.exclude):
        return 0.0
    # Pages on the official domain are about the entity by definition.
    if entity.official_domain and (doc.meta.domain == entity.official_domain or doc.meta.domain.endswith("." + entity.official_domain)):
        return 1.0
    squash = re.sub(r"[^a-z0-9]", "", text)
    if re.sub(r"[^a-z0-9]", "", entity.name.lower()) not in squash:
        return 0.0
    anchors, hits = 0, 0
    if entity.official_domain:
        anchors += 1
        hits += doc.meta.domain == entity.official_domain or doc.meta.domain.endswith("." + entity.official_domain)
    for value in (entity.country, entity.city):
        if value:
            anchors += 1
            hits += value.lower() in text
    if entity.industry_keywords:
        anchors += 1
        hits += any(k.lower() in text for k in entity.industry_keywords)
    if anchors == 0:
        return 0.5
    return round(hits / anchors, 2)


def completeness_score(ev: Evidence) -> float:
    s = 0.0
    s += 0.5 if claim_date(ev) else 0.0
    s += 0.25 if len(ev.claim.evidence_snippet) >= 30 else 0.1
    s += 0.25 if ev.source.publisher else 0.0
    return round(s, 2)


def score_evidence(ev: Evidence, doc: Document, entity: EntityAnchor, as_of: date) -> Evidence:
    parts = {
        "authority": authority_score(ev.claim.field, ev.source.source_type),
        "freshness": freshness_score(ev.claim.field, claim_date(ev), as_of),
        "identity": identity_score(doc, entity),
        "completeness": completeness_score(ev),
    }
    ev.score_breakdown = parts
    ev.identity_match = parts["identity"]
    ev.freshness = parts["freshness"]
    ev.reliability = round(sum(WEIGHTS[k] * v for k, v in parts.items()), 3)
    if ev.identity_match < 0.5:
        ev.accepted = False
        ev.flags.append("identity_mismatch")
    if "prompt_injection_suspected" in doc.flags:
        ev.accepted = False
        ev.flags.append("quarantined_source")
    return ev


# ---------------------------------------------------------------------------
# Copy detection: shingles + Jaccard similarity
# ---------------------------------------------------------------------------
def shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9$%.]+", text.lower())
    return {" ".join(words[i: i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def find_copies(docs: list[Document], threshold: float = 0.8) -> dict[str, str]:
    """Map doc_id -> origin doc_id for pages that are near-copies of another page.

    The origin is the higher-authority page (a press release beats a syndicated copy);
    ties go to the earlier publication.
    """
    origin: dict[str, str] = {}
    sh = {d.meta.doc_id: shingles(d.text) for d in docs}
    ordered = sorted(docs, key=lambda d: (-AUTHORITY.get(d.meta.source_type, 0.2), d.meta.publication_date or date.max))
    for i, a in enumerate(ordered):
        if a.meta.doc_id in origin:
            continue
        for b in ordered[i + 1:]:
            if b.meta.doc_id not in origin and jaccard(sh[a.meta.doc_id], sh[b.meta.doc_id]) >= threshold:
                origin[b.meta.doc_id] = a.meta.doc_id
    return origin
