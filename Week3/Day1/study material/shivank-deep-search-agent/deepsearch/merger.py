"""
merger.py — from a pile of evidence to one clean, traceable answer.

Steps (in this order — deduplicate BEFORE resolving conflicts):

1. Normalise values ("Series B — $40M" == "series b $40 million").
2. Cluster evidence for the same field + same (or near-same) value.
3. Collapse copies: evidence from syndicated copies counts once (origin_id).
4. Score each cluster: confidence = 1 - Π(1 - reliability_i) over INDEPENDENT origins
   (a "noisy-OR": two independent decent sources beat one great source).
5. Resolve conflicts between clusters of the same field:
   accepted_one / rejected_weak / time_separated / unresolved.
6. Enforce the high-stakes rule: regulatory or security claims need a regulator/
   official source or two independent sources, otherwise confidence is capped.
7. Keep provenance: every merged claim lists the evidence ids and URLs behind it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from .schemas import ConflictRecord, Evidence, MergedClaim
from .validator import HIGH_STAKES_FIELDS, TIME_SENSITIVE_FIELDS, claim_date

STRONG = 0.60   # a cluster at or above this confidence is "trustworthy"
WEAK = 0.40     # below this we reject a value if a strong alternative exists


def normalize_value(value: str) -> str:
    v = value.lower()
    v = v.replace("million", "m").replace("authorization", "authorisation").replace("license", "licence")
    v = re.sub(r"\b(ms|mr|dr)\.?\s+", "", v)
    v = re.sub(r"[^a-z0-9.%/ ]", " ", v)
    v = re.sub(r"(\d)\s+(m|bn|k|%)\b", r"\1\2", v)  # "40 m" -> "40m"
    return re.sub(r"\s+", " ", v).strip()


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def same_value(a: str, b: str, threshold: float) -> bool:
    """Exact match, near-duplicate (token overlap) or containment ('ananya rao' in 'ananya rao since 2026')."""
    if a == b or token_jaccard(a, b) >= threshold:
        return True
    ta, tb = set(a.split()), set(b.split())
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(small) >= 2 and small <= big


def cluster_values(evs: list[Evidence], threshold: float = 0.6) -> list[list[Evidence]]:
    """Greedy clustering: exact normalised match first, then near-duplicates by token overlap."""
    clusters: list[tuple[str, list[Evidence]]] = []
    for ev in sorted(evs, key=lambda e: -e.reliability):
        key = normalize_value(ev.claim.value)
        for ckey, members in clusters:
            if same_value(key, ckey, threshold):
                members.append(ev)
                break
        else:
            clusters.append((key, [ev]))
    return [members for _, members in clusters]


def _cluster_to_claim(field: str, members: list[Evidence], high_stakes_rule: bool = True) -> MergedClaim:
    best = max(members, key=lambda e: e.reliability)
    # one vote per independent origin: keep the most reliable piece of evidence per origin
    per_origin: dict[str, float] = {}
    for e in members:
        origin = e.origin_id or e.source.doc_id
        per_origin[origin] = max(per_origin.get(origin, 0.0), e.reliability)
    disbelief = 1.0
    for r in per_origin.values():
        disbelief *= (1 - r)
    confidence = min(0.99, 1 - disbelief)
    dates = [d for d in (claim_date(e) for e in members) if d]
    if high_stakes_rule and field in HIGH_STAKES_FIELDS:
        strong_source = any(e.source.source_type in ("regulator", "official") for e in members)
        if not strong_source and len(per_origin) < 2:
            confidence = min(confidence, 0.35)
    return MergedClaim(
        field=field, value=best.claim.value, as_of=max(dates) if dates else None,
        confidence=round(confidence, 3), status="unknown",
        supporting_sources=sorted({e.source.url for e in members}),
        independent_sources=len(per_origin), evidence_ids=[e.id for e in members],
    )


def merge_field(field: str, evs: list[Evidence], dedup_threshold: float = 0.6,
                high_stakes_rule: bool = True) -> tuple[list[MergedClaim], ConflictRecord | None]:
    evs = sorted(evs, key=lambda e: (e.source.doc_id, e.claim.value))  # deterministic order => repeatable results
    claims = [_cluster_to_claim(field, members, high_stakes_rule) for members in cluster_values(evs, dedup_threshold)]
    if len(claims) == 1:
        claims[0].status = "current"  # "current" = the value we report for this field
        return claims, None

    candidates = [{"value": c.value, "confidence": c.confidence, "as_of": c.as_of.isoformat() if c.as_of else None,
                   "independent_sources": c.independent_sources} for c in claims]
    strong = [c for c in claims if c.confidence >= STRONG]
    weak = [c for c in claims if c.confidence < WEAK]

    if not strong:
        best = max(claims, key=lambda c: c.confidence)
        for c in claims:
            c.status = "unknown"
        return claims, ConflictRecord(field=field, candidates=candidates, resolution="unresolved", chosen_value=None,
                                      rationale=f"No value reaches confidence {STRONG}; best guess '{best.value}' "
                                                f"({best.confidence:.2f}) is reported as uncertain.")

    if field in TIME_SENSITIVE_FIELDS:
        winner = max(strong, key=lambda c: (c.as_of or date.min, c.confidence))
        same_time_rivals = [c for c in strong if c is not winner and c.as_of and winner.as_of
                            and abs((winner.as_of - c.as_of).days) <= 30]
        if same_time_rivals:
            for c in claims:
                c.status = "unknown"
            return claims, ConflictRecord(field=field, candidates=candidates, resolution="unresolved", chosen_value=None,
                                          rationale="Strong sources from the same period disagree; reported both, flagged for a human.")
        winner.status = "current"
        older_ok = [c for c in claims if c is not winner and c not in weak]
        for c in claims:
            if c is not winner:
                c.status = "historical" if c in older_ok and c.as_of and winner.as_of and c.as_of < winner.as_of else "unknown"
        rejected = [c.value for c in claims if c is not winner and c.status != "historical"]
        kept_hist = [c.value for c in claims if c.status == "historical"]
        resolution = "time_separated" if kept_hist else "rejected_weak"
        parts = [f"'{winner.value}' is the newest strong value (as of {winner.as_of})."]
        if kept_hist:
            parts.append(f"Older values kept as history: {', '.join(repr(v) for v in kept_hist)}.")
        if rejected:
            parts.append(f"Rejected weak or unreliable values: {', '.join(repr(v) for v in rejected)}.")
        return claims, ConflictRecord(field=field, candidates=candidates, resolution=resolution,
                                      chosen_value=winner.value, rationale=" ".join(parts))

    winner = max(claims, key=lambda c: c.confidence)
    winner.status = "current"
    return claims, ConflictRecord(field=field, candidates=candidates, resolution="accepted_one", chosen_value=winner.value,
                                  rationale=f"Kept the highest-confidence value '{winner.value}' ({winner.confidence:.2f}).")


def merge_all(evidence: list[Evidence], dedup_threshold: float = 0.6,
              high_stakes_rule: bool = True) -> tuple[dict[str, list[MergedClaim]], list[ConflictRecord]]:
    by_field: dict[str, list[Evidence]] = defaultdict(list)
    for ev in evidence:
        if ev.accepted:
            by_field[ev.claim.field].append(ev)
    merged, conflicts = {}, []
    for field, evs in by_field.items():
        claims, conflict = merge_field(field, evs, dedup_threshold, high_stakes_rule)
        merged[field] = sorted(claims, key=lambda c: (c.status != "current", -c.confidence))
        if conflict:
            conflicts.append(conflict)
    return merged, conflicts


def confidence_label(x: float) -> str:
    return "high" if x >= 0.8 else "medium" if x >= 0.55 else "low"
