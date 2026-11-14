"""
LAB 4 — The executor: workers, evidence objects and parallel fan-out
================================================================================
Course module: 4 (Executor and tool layer) + 9 (Cost, latency and scale)

What you will see
  1. ONE worker handling ONE sub-question: queries → search results (with metadata) → fetched
     pages → security scan → extracted claims → Evidence objects (typed, with provenance).
  2. The same 8 workers run one-after-another vs in parallel, with a simulated 0.15 s network
     delay per tool call, so you can measure the speed-up of the orchestrator-worker pattern.

Run it
  python labs/lab04_executor_parallel.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

import json
import os
import time

from deepsearch.budget import Budget
from deepsearch.config import describe
from deepsearch.executor import Executor
from deepsearch.llm import LLM
from deepsearch.orchestrator import DEFAULT_QUESTION, default_entity
from deepsearch.planner import create_plan
from deepsearch.tools import search_web


def main() -> None:
    banner("LAB 4 — Executor workers and parallel fan-out")
    print("  Settings:", describe())
    entity = default_entity()
    plan = create_plan(LLM(budget=Budget()), DEFAULT_QUESTION, entity)
    sq = next(q for q in plan.sub_questions if q.id == "q2")  # "Who is the current CEO...?"

    step(f"1. One worker for {sq.id}: {sq.question}")
    ex = Executor(entity, Budget(), max_parallel=1, results_per_query=3)
    evidence = ex.run_worker(sq, iteration=1)
    print("  Queries written:", ex.queries_used[sq.id])
    results = search_web(ex.queries_used[sq.id][0], max_results=3)
    table([{"rank": r.rank, "domain": r.domain, "published": r.published_date, "title": r.title} for r in results],
          max_width=60)
    print(f"\n  Pages fetched: {len(ex.docs)} | evidence objects: {len(evidence)}")
    print("  One Evidence object (this is what flows to the validator):")
    print("  " + json.dumps(evidence[0].model_dump(mode="json"), indent=2).replace("\n", "\n  ")[:1400])
    pause_note("Every claim carries its source URL, source type and dates. identity_match and reliability "
               "are still 0 here — the validator fills them in (Lab 6). Without provenance it couldn't.")

    step("2. Sequential vs parallel workers (simulated 0.15 s per tool call)")
    os.environ["DS_SIMULATED_LATENCY"] = "0.15"
    timings = {}
    for mode, parallel, workers in (("sequential", False, 1), ("parallel x4", True, 4), ("parallel x8", True, 8)):
        ex = Executor(entity, Budget(), max_parallel=workers, results_per_query=3)
        t0 = time.time()
        evs = ex.run(plan.sub_questions, iteration=1, parallel=parallel)
        timings[mode] = time.time() - t0
        print(f"  {mode:<12} {timings[mode]:5.2f} s   ({len(evs)} evidence, {ex.budget.tool_calls} tool calls)")
    os.environ["DS_SIMULATED_LATENCY"] = "0"
    speedup = timings["sequential"] / timings["parallel x8"]
    pause_note(f"Same work, ~{speedup:.1f}x faster with 8 parallel workers. Tokens and tool calls stay the same: "
               "parallelism buys TIME, not money. (Anthropic reported up to 90% less research time this way.)")


if __name__ == "__main__":
    main()
