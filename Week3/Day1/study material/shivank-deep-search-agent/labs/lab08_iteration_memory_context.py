"""
LAB 8 — Iteration, context management and long-term memory
================================================================================
Course module: 7 (Context, memory and iteration)

What you will see
  1. The iterative loop on a TIGHT reading budget (top-1 result per query):
     iteration 1 leaves a high-stakes gap → the evaluator says CONTINUE → iteration 2 uses a
     different query angle, finds the official security notice → STOP.
  2. Context management: why the planner sees compact evidence objects, not raw pages.
  3. Long-term memory: run the agent twice. Run 1 learns lessons; run 2 reads them, skips
     known-bad domains and spends fewer tool calls.

Run it
  python labs/lab08_iteration_memory_context.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

import json

from deepsearch.budget import estimate_tokens
from deepsearch.config import SETTINGS, describe
from deepsearch.memory import Memory
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity


def main() -> None:
    banner("LAB 8 — Iteration, context and memory")
    print("  Settings:", describe())

    step("1. Iterative loop with a tight budget (read only the top result per query)")
    SETTINGS.results_per_query = 1
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False)
    agent.run()
    table([{"iteration": it["iteration"], "new_evidence": it["new_evidence"],
            "decision": it["decision"]["action"], "reason": it["decision"]["reasons"][0]}
           for it in agent.iterations], max_width=70)
    print("  Security-incident evidence, in the order it was found:")
    for e in agent.evidence:
        if e.claim.field == "security_incident":
            print(f"    {e.id} [{e.source.source_type}] {e.source.url}")
    pause_note("After iteration 1 the security incident had ONE news source → capped at low confidence by the "
               "high-stakes rule. The evaluator asked for more; iteration 2 searched a new angle ('security "
               "notice') and found the official notice. The system searched BETTER, not just more.")

    step("2. Context management: what goes into the model's context window?")
    SETTINGS.results_per_query = 3
    SETTINGS.verbose = False
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False)
    report = agent.run()
    from deepsearch.tools import corpus_docs
    pages_read = [d for d in corpus_docs() if any(e.source.doc_id == d["id"] for e in agent.evidence)]
    raw_tokens = sum(estimate_tokens(d["text"]) for d in pages_read) * 6  # real pages are ~6x longer than ours
    evidence_tokens = estimate_tokens(json.dumps([{"f": e.claim.field, "v": e.claim.value, "src": e.source.doc_id,
                                                   "rel": e.reliability, "date": str(e.source.publication_date)}
                                                  for e in agent.evidence if e.accepted]))
    state_tokens = estimate_tokens(json.dumps([{"field": f.field, "value": f.value, "conf": f.confidence}
                                               for f in report.key_findings]))
    table([{"what the planner could see": "raw page text (realistic size)", "≈ tokens": f"{raw_tokens:,}"},
           {"what the planner could see": "compact evidence objects", "≈ tokens": f"{evidence_tokens:,}"},
           {"what the planner could see": "merged state (one line per field)", "≈ tokens": f"{state_tokens:,}"}],
          max_width=40)
    pause_note("Raw text stays OUTSIDE the prompt (on disk / in Python objects). Each layer passes a compact, "
               "structured summary forward. This prevents context overload, repetition and lost decisions.")

    step("3. Long-term memory across two runs")
    mem = Memory()
    mem.reset()
    stats = []
    for run in (1, 2):
        agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), memory=Memory(), use_memory=True)
        r = agent.run()
        stats.append({"run": run, "tool_calls": r.run_stats["tool_calls"], "pages_read": r.run_stats["pages_read"],
                      "skipped_by_memory": len(agent.executor.skipped),
                      "correct_ceo": next(f.value for f in r.key_findings if f.field == "ceo")})
    table(stats)
    print("\n  Lessons stored in memory/lessons.json (read by the planner next time):")
    for lesson in Memory().data["lessons"]:
        print(f"    • {lesson['text']}  (seen {lesson['times_seen']}x)")
    pause_note("Memory stores reusable LESSONS and source track records — not raw personal data — and it never "
               "overrides fresh evidence: facts are still re-verified every run.")


if __name__ == "__main__":
    main()
