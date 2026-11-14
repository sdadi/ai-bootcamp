"""
LAB 3 — The planning layer: Plan Creator → Plan Reviewer → Plan Evaluator
================================================================================
Course module: 3 (Planning layer)

What you will see
  1. Plan Creator turns one vague question into typed sub-questions (5 lenses).
  2. Plan Reviewer catches a WEAK plan's problems (no identity check, high-stakes not marked,
     no stop rule) and fixes them.
  3. Plan Evaluator decides CONTINUE or STOP for four different situations.

Run it
  python labs/lab03_planner.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

from deepsearch.budget import Budget
from deepsearch.config import describe
from deepsearch.llm import LLM
from deepsearch.planner import create_plan, evaluate_progress, review_plan
from deepsearch.schemas import MergedClaim, ResearchPlan, SubQuestion
from deepsearch.orchestrator import DEFAULT_QUESTION, default_entity


def merged_state(confidences: dict[str, float]) -> dict[str, list[MergedClaim]]:
    return {f: [MergedClaim(field=f, value="x", confidence=c, status="current", supporting_sources=[],
                            independent_sources=1, evidence_ids=[])] for f, c in confidences.items()}


def main() -> None:
    banner("LAB 3 — Planner multi-agent system")
    print("  Settings:", describe())
    budget = Budget()
    llm = LLM(budget=budget)
    entity = default_entity()

    step("1. Plan Creator")
    plan = create_plan(llm, DEFAULT_QUESTION, entity)
    table([{"id": q.id, "lens": q.lens, "priority": q.priority, "high_stakes": q.high_stakes,
            "fresh_days": q.freshness_days, "question": q.question} for q in plan.sub_questions],
          ["id", "lens", "priority", "high_stakes", "fresh_days", "question"], max_width=70)
    print("\n  Source strategy:", *plan.source_strategy, sep="\n    • ")
    print("  Stop criteria:", *plan.stop_criteria, sep="\n    • ")

    step("2. Plan Reviewer on a deliberately WEAK plan")
    weak = ResearchPlan(
        goal=DEFAULT_QUESTION, entity=entity, stop_criteria=[], validation_requirements=[],
        sub_questions=[
            SubQuestion(id="q1", question="NimbusPay licence?", fields=["regulatory_status"], preferred_sources=[]),
            SubQuestion(id="q2", question="Who runs NimbusPay right now and since when?", fields=["ceo"],
                        preferred_sources=["official"]),
        ],
    )
    fixed, review = review_plan(weak)
    print("  Issues found:", *review.issues, sep="\n    ✗ ")
    print("  Fixes applied:", *review.fixes_applied, sep="\n    ✓ ")
    print(f"  Approved: {review.approved}  (vague questions must be rewritten by the Creator — rules can't guess intent)")

    step("3. Plan Evaluator: continue or stop?")
    unresolved: set[str] = set()
    all_good = {f: 0.9 for q in plan.sub_questions for f in q.fields}
    scenarios = [
        ("Iteration 1, licence + security still weak", {**all_good, "regulatory_status": 0.35, "security_incident": 0.3}, 1, 5),
        ("Iteration 2, everything answered", all_good, 2, 3),
        ("Iteration 2, gaps remain but no new evidence", {**all_good, "pricing_domestic_cards": 0.3}, 2, 0),
        ("Iteration 3 = max iterations", {**all_good, "uptime_90d": 0.4}, 3, 2),
    ]
    rows = []
    for name, conf, it, new in scenarios:
        d = evaluate_progress(plan, merged_state(conf), unresolved, it, 3, new, budget)
        rows.append({"situation": name, "decision": d.action.upper(), "why": d.reasons[0]})
    table(rows, ["situation", "decision", "why"], max_width=62)
    pause_note("Stopping rules prevent two failures: stopping after one plausible page, and searching forever. "
               "Next: Lab 4 runs the executor workers in parallel.")


if __name__ == "__main__":
    main()
