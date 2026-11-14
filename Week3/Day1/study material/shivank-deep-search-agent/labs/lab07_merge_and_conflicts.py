"""
LAB 7 — Merging: deduplication, conflict resolution, provenance, structured output
================================================================================
Course module: 6 (Merging and final answer construction)

What you will see
  1. Deduplication of messy value variants, and what goes wrong with a threshold that is too
     low (OVER-deduplication merges different facts) or too high (UNDER-deduplication).
  2. Every conflict the agent found and how it was resolved:
     accepted_one / rejected_weak / time_separated / unresolved.
  3. Provenance: from one final claim back to the exact sentences that support it.
  4. The final structured output vs a raw evidence dump (size and usefulness).

Run it
  python labs/lab07_merge_and_conflicts.py
"""

from _common import banner, pause_note, step, table, wrap  # noqa: E402

import json

from deepsearch.budget import estimate_tokens
from deepsearch.config import SETTINGS, describe
from deepsearch.merger import cluster_values, normalize_value
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity
from deepsearch.schemas import Evidence, ExtractedClaim, SourceMeta


def fake_evidence(values: list[str]) -> list[Evidence]:
    """Value variants like the ones real LLM extraction produces from different pages."""
    meta = SourceMeta(doc_id="x", url="https://x.example", domain="x.example", title="x", source_type="news",
                      retrieval_date=SETTINGS.as_of)
    return [Evidence(id=f"v{i}", sub_question_id="q", source=meta.model_copy(update={"doc_id": f"x{i}"}),
                     claim=ExtractedClaim(field="ceo" if "rao" in v.lower() or "mehra" in v.lower() else "latest_funding",
                                          value=v, evidence_snippet="(synthetic example)"), reliability=0.7)
            for i, v in enumerate(values)]


def main() -> None:
    banner("LAB 7 — Merge, deduplicate, resolve conflicts, keep provenance")
    print("  Settings:", describe())

    step("1. Deduplication: normalise, then cluster near-duplicates")
    variants = ["Ananya Rao", "Ms. Ananya Rao", "ananya rao (CEO since January 2026)", "Rohit Mehra",
                "Series B — $40M", "Series B, $40 million", "series b $40 m", "Series A — $8M"]
    for v in variants:
        print(f"  {v!r:<42} → normalised {normalize_value(v)!r}")
    rows = []
    for threshold in (0.15, 0.6, 1.01):
        ceo = cluster_values([e for e in fake_evidence(variants) if e.claim.field == "ceo"], threshold)
        fund = cluster_values([e for e in fake_evidence(variants) if e.claim.field == "latest_funding"], threshold)
        rows.append({"threshold": threshold, "ceo clusters": len(ceo), "funding clusters": len(fund),
                     "funding groups": " | ".join("+".join(e.claim.value for e in c) for c in fund)})
    table(rows, max_width=80)
    pause_note("0.15 = OVER-dedup: Series A and Series B collapse into one 'fact' (wrong!). 1.01 = exact match only: "
               "still works here thanks to normalisation + containment, but real-world variants slip through. "
               "0.6 is our default. Never collapse genuinely different claims.")

    step("2. Run the agent and inspect the conflicts")
    SETTINGS.verbose = False
    SETTINGS.results_per_query = 5
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False)
    report = agent.run()
    for c in report.conflicts:
        print(f"\n  {c.field} → {c.resolution.upper()}")
        table([{"value": x["value"], "confidence": x["confidence"], "as_of": x["as_of"],
                "independent_sources": x["independent_sources"]} for x in c.candidates])
        print(wrap("why: " + c.rationale, indent="    "))

    step("3. Provenance: trace one final claim back to its sentences")
    finding = next(f for f in report.key_findings if f.field == "regulatory_status")
    print(f"  FINAL: regulatory_status = {finding.value!r} ({finding.confidence}) sources {finding.sources}")
    for e in agent.evidence:
        if e.claim.field == "regulatory_status" and e.claim.value == finding.value and e.accepted:
            print(f"    ← {e.id} [{e.source.source_type}] {e.source.url}")
            print(wrap(f"\"{e.claim.evidence_snippet}\"", indent="        "))

    step("4. Clean structured result vs raw evidence dump")
    raw = json.dumps([e.model_dump(mode="json") for e in agent.evidence])
    clean = report.model_dump_json(include={"answer", "confidence", "key_findings", "conflicts", "open_questions"})
    print(f"  raw evidence dump : {len(agent.evidence)} items, ~{estimate_tokens(raw):,} tokens")
    print(f"  final report      : {len(report.key_findings)} findings, ~{estimate_tokens(clean):,} tokens")
    print("  final report shape:", list(json.loads(report.model_dump_json()).keys()))
    pause_note("The goal is not to keep every claim; it is to build the best final representation — with "
               "confidence, dates, conflicts and sources still attached.")


if __name__ == "__main__":
    main()
