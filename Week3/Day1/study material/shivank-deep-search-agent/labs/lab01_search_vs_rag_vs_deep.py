"""
LAB 1 — Normal search vs basic RAG vs deep search (same question, three systems)
================================================================================
Course module: 1 (Framing the problem)

What you will see
  The same due-diligence question goes through:
    A. Naive search  — search once, read the top 3 pages, take what they say.
    B. Basic RAG     — retrieve the top chunks for each topic, take what they say.
    C. Deep search   — plan → parallel workers → validate → merge → report.
  Then all three are compared with the hand-checked answers in data/gold_set.json.

Run it (from the project root folder)
  python labs/lab01_search_vs_rag_vs_deep.py

Things to notice
  * Naive search believes a poisoned SEO page ("licence revoked") and an old CEO.
  * Basic RAG answers with the 2022 Series A because old chunks rank well.
  * Deep search gets every field right AND says which values are history.
"""

from _common import banner, pause_note, step, table  # noqa: E402  (must be first: sets sys.path)

import json

from deepsearch.baselines import basic_rag_answer, naive_search_answer
from deepsearch.config import DATA_DIR, describe
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity


def main() -> dict:
    banner("LAB 1 — Search vs RAG vs Deep Search")
    print("  Settings:", describe())
    print("  Question:", DEFAULT_QUESTION)

    step("A. Naive search (one query, top 3 pages, no validation)")
    naive = naive_search_answer(DEFAULT_QUESTION)
    print("  Pages used:", *naive["sources"], sep="\n    ")

    step("B. Basic RAG (top-2 chunks per topic from a fixed index)")
    rag = basic_rag_answer(DEFAULT_QUESTION)
    print(f"  Chunks came from {len(rag['sources'])} pages")

    step("C. Deep search (full multi-agent pipeline)")
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False)
    report = agent.run()
    deep = {f.field: f.value for f in report.key_findings}

    step("Comparison with the gold answers")
    gold = json.loads((DATA_DIR / "gold_set.json").read_text(encoding="utf-8"))["cases"][0]["expected"]
    rows, score = [], {"naive": 0, "rag": 0, "deep": 0}
    for field, exp in gold.items():
        row = {"field": field, "gold": exp["value"]}
        for name, ans in (("naive", naive["answers"]), ("rag", rag["answers"]), ("deep", deep)):
            got = ans.get(field, "—")
            ok = got == exp["value"]
            score[name] += ok
            row[name] = ("✓ " if ok else "✗ ") + got
        rows.append(row)
    table(rows, ["field", "gold", "naive", "rag", "deep"], max_width=30)
    n = len(gold)
    print(f"\n  Correct fields → naive: {score['naive']}/{n} | RAG: {score['rag']}/{n} | deep search: {score['deep']}/{n}")

    step("What deep search knew that the others did not")
    for h in report.historical_findings:
        print(f"  • {h.field}: '{h.value}' is HISTORY (as of {h.as_of}), not the current value")
    for note in report.security_notes:
        print(f"  • {note}")

    pause_note("Open the run folder printed above and read report.md — that is what a 'validated, merged, "
               "structured' answer looks like. Then continue to Lab 2.")
    return score


if __name__ == "__main__":
    main()
