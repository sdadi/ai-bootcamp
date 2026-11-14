"""
LAB 6 — Validation: reliability scoring, identity, copies and time
================================================================================
Course module: 5 (Validation, reliability and temporal reasoning)

What you will see
  1. Every piece of evidence scored on authority, freshness, identity and completeness.
  2. The look-alike company (Nimbus Pay Ltd, UK payroll) rejected by identity matching.
  3. A syndicated press release detected as a COPY (5 copies = 1 origin, not 5 confirmations).
  4. Temporal reasoning: the four dates, and how freshness decays with a field's half-life.

Run it
  python labs/lab06_validation.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

from datetime import date

from deepsearch.config import SETTINGS, describe
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity
from deepsearch.validator import HALF_LIFE_DAYS, WEIGHTS, freshness_score, jaccard, shingles


def main() -> None:
    banner("LAB 6 — Validation, reliability and temporal reasoning")
    print("  Settings:", describe())
    print("  reliability = " + " + ".join(f"{w}*{k}" for k, w in WEIGHTS.items()))

    SETTINGS.verbose = False  # keep the agent quiet; we only want its evidence
    SETTINGS.results_per_query = 5  # read a bit more so the copied press release shows up
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False)
    agent.run()

    step("1. Reliability scores for every piece of evidence (sorted)")
    rows = []
    for e in sorted(agent.evidence, key=lambda e: -e.reliability):
        b = e.score_breakdown
        rows.append({"field": e.claim.field, "value": e.claim.value, "source": e.source.source_type,
                     "auth": b["authority"], "fresh": b["freshness"], "ident": b["identity"],
                     "compl": b["completeness"], "RELIAB": e.reliability,
                     "ok": "✓" if e.accepted else "✗ " + ",".join(e.flags)})
    table(rows, max_width=26)

    step("2. Identity: same name, different company")
    for e in agent.evidence:
        if "identity_mismatch" in e.flags:
            print(f"  ✗ {e.source.url} → '{e.claim.field} = {e.claim.value}' rejected (identity {e.identity_match})")
    pause_note("Look at table 1: the UK page scores ~0.8 reliability (an official site, fresh, complete) and is "
               "STILL rejected. Identity is a gate, not just a weight — a perfect source about the wrong company "
               "is worthless. Without anchors the agent would report 'CEO Daniel Hughes, HQ Manchester'.")

    step("3. Copies are not independent confirmations")
    from deepsearch.tools import corpus_docs
    texts = {d["id"]: d for d in corpus_docs()}
    a, b, c = texts["d05"], texts["d06"], texts["d07"]
    print(f"  press release  vs syndicated copy : Jaccard = {jaccard(shingles(a['text']), shingles(b['text'])):.2f}  → COPY")
    print(f"  press release  vs BizDaily report : Jaccard = {jaccard(shingles(a['text']), shingles(c['text'])):.2f}  → independent")
    for e in agent.evidence:
        if "copy_of_other_page" in e.flags:
            print(f"  {e.source.url} → origin {e.origin_id} (its vote is merged with the original)")

    step("4. Four dates — and why they differ")
    table([{"page": texts[i]["title"][:44], "published": texts[i]["published_date"],
            "last_updated": texts[i]["last_updated"], "event_date_in_text": ev, "retrieved": SETTINGS.as_of}
           for i, ev in (("d03", "2025-08-14 (final auth.)"), ("d07", "2022 (Series A mention)"),
                         ("d11", "—"), ("d09", "2026-01-12 (outage)"))], max_width=44)

    step("5. Freshness decay by field (claim dated 2026-04-01, viewed on different days)")
    rows = []
    for as_of in (date(2026, 4, 1), date(2026, 7, 1), date(2026, 11, 1), date(2027, 11, 1)):
        rows.append({"viewed_on": as_of, **{f"{f} (½ life {HALF_LIFE_DAYS[f]}d)": freshness_score(f, date(2026, 4, 1), as_of)
                                            for f in ("uptime_90d", "pricing_domestic_cards", "ceo", "founded_year")}})
    table(rows, max_width=34)
    pause_note("Same date, different meaning: a 7-month-old uptime number is stale, a 7-month-old CEO fact is "
               "still decent, and the founding year never expires. Freshness must be field-specific.")


if __name__ == "__main__":
    main()
