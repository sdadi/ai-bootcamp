"""
LAB 11 — Evaluating a deep-search agent end to end
================================================================================
Course module: 10 (Evaluation and failure modes)

What you will see
  1. A gold set (data/gold_set.json): hand-checked answers, forbidden content, expected conflicts.
  2. Four systems scored on the same case:
       naive search | basic RAG | deep search WITHOUT validation (ablation) | full deep search
  3. Programmatic metrics: field accuracy, history recall, citation support, conflict recall,
     forbidden content, over-confident errors, cost.
  4. An LLM-as-judge rubric (1–5 on 8 dimensions) + "main failure → recommended fix".

Run it
  python labs/lab11_evaluation.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

import json

from deepsearch.baselines import basic_rag_answer, naive_search_answer
from deepsearch.budget import Budget
from deepsearch.config import DATA_DIR, SETTINGS, describe
from deepsearch.evaluation import RUBRIC, judge, score_answers
from deepsearch.llm import LLM
from deepsearch.orchestrator import DeepSearchAgent
from deepsearch.schemas import EntityAnchor


def main() -> None:
    banner("LAB 11 — Evaluation: gold set, metrics, ablation, LLM-as-judge")
    print("  Settings:", describe())
    SETTINGS.verbose = False
    case = json.loads((DATA_DIR / "gold_set.json").read_text(encoding="utf-8"))["cases"][0]
    print(f"  Gold case: {case['id']} — {len(case['expected'])} fields, must not mention {case['must_not_appear']}")

    step("1. Run four systems on the same question")
    results = {}
    naive = naive_search_answer(case["question"])
    results["naive search"] = (naive["answers"], None, None)
    rag = basic_rag_answer(case["question"])
    results["basic RAG"] = (rag["answers"], None, None)
    for name, validation in (("deep, no validation", False), ("deep search (full)", True)):
        agent = DeepSearchAgent(case["question"], EntityAnchor(**case["entity"]), use_memory=False, validation=validation)
        rep = agent.run()
        results[name] = ({f.field: f.value for f in rep.key_findings}, rep, rep.run_stats)

    step("2. Programmatic metrics")
    rows, all_metrics = [], {}
    for name, (answers, rep, stats) in results.items():
        m = score_answers(answers, case, rep, stats)
        all_metrics[name] = m
        rows.append({"system": name, "accuracy": m["field_accuracy"], "history": m.get("history_recall", "—"),
                     "citations": m.get("citation_support", "—"), "conflicts": m.get("conflict_recall", "—"),
                     "forbidden": ", ".join(m["forbidden_content"]) or "none",
                     "overconf.": ", ".join(m.get("overconfident_errors", [])) or "—",
                     "tool_calls": m.get("tool_calls", "—")})
    table(rows, max_width=28)
    print("\n  Wrong fields per system:")
    for name, m in all_metrics.items():
        print(f"    {name:<22} {m['wrong_fields'] or 'none'}")
    pause_note("The ablation shows WHY each layer exists. Without validation, the injected 'licence revoked' page "
               "looks as trustworthy as the regulator, so the licence status ends up UNRESOLVED; and the UK "
               "look-alike's CEO 'Daniel Hughes' leaks into the report as a 'former CEO'. Same pipeline, same "
               "pages — only the validation layer removed. Ablations are the clearest way to justify a component.")

    step("3. LLM-as-judge rubric (mock = rule-based stand-in; set a real LLM to use a model judge)")
    llm = LLM(budget=Budget())
    rows = []
    for name, (answers, rep, _stats) in results.items():
        js = judge(llm, case, rep or answers, all_metrics[name])
        rows.append({"system": name, **{k[:10]: js.scores.get(k, "-") for k in RUBRIC},
                     "avg": round(sum(js.scores.values()) / len(js.scores), 2), "fix": js.recommended_fix})
    table(rows, max_width=46)
    pause_note("Use the judge for what code can't measure, calibrate it against a few human-scored reports, and "
               "keep the programmatic metrics as the backbone. Turn every failure into a test case in the gold set.")


if __name__ == "__main__":
    main()
