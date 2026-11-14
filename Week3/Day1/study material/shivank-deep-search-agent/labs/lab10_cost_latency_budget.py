"""
LAB 10 — Cost, latency and budgets
================================================================================
Course module: 9 (Cost, latency and scaling) — NEW topic

What you will see
  1. Where the tokens go, stage by stage (planning, query writing, extraction, synthesis).
  2. Why multi-agent research costs much more than chat (token multiplier).
  3. Model routing: the same run priced with a large model everywhere vs a small model for extraction.
  4. A hard budget: the agent stops cleanly when its tool-call budget runs out and reports the gaps.
  5. A latency model you can play with: sequential vs parallel workers.

Prices are EXAMPLE numbers (see deepsearch/budget.py). Always check your provider's pricing page.

Run it
  python labs/lab10_cost_latency_budget.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

from deepsearch.budget import PRICE_PER_MTOK, Budget
from deepsearch.config import SETTINGS, describe
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity


def cost(tokens_in: int, tokens_out: int, tier: str) -> float:
    p_in, p_out = PRICE_PER_MTOK[tier]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def main() -> None:
    banner("LAB 10 — Cost, latency and budgets")
    print("  Settings:", describe())
    SETTINGS.verbose = False

    step("1. Where do the tokens go?")
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False)
    report = agent.run()
    stats = report.run_stats
    table([{"stage": k, "llm_calls": int(v["llm_calls"]), "tool_calls": int(v["tool_calls"]), "tokens": int(v["tokens"]),
            "est_usd": f"{v['usd']:.5f}"} for k, v in stats["by_stage"].items()])
    print(f"\n  TOTAL: {stats['llm_calls']} LLM calls, {stats['tool_calls']} tool calls, "
          f"{stats['input_tokens'] + stats['output_tokens']:,} tokens, ≈ ${stats['estimated_usd']:.4f}")
    pause_note("Mock mode counts tokens of our SHORT fictional pages. Real pages are 5-20x longer, so expect "
               "real runs to cost proportionally more. Extraction (reading pages) dominates — optimise there first.")

    step("2. The token multiplier: chat vs agent vs multi-agent")
    chat_tokens = 1_500
    total = stats["input_tokens"] + stats["output_tokens"]
    table([{"system": "one chat answer", "≈ tokens": f"{chat_tokens:,}", "x chat": "1x"},
           {"system": "this multi-agent run (mock pages)", "≈ tokens": f"{total:,}", "x chat": f"{total / chat_tokens:.0f}x"},
           {"system": "Anthropic's report (real research)", "≈ tokens": "—", "x chat": "~4x single agent, ~15x multi-agent"}])
    pause_note("Multi-agent systems win on hard, broad questions because they spend more tokens in parallel. "
               "Use them when the value of the answer justifies the bill; use one agent for simple lookups.")

    step("3. Model routing: which model does which job?")
    tin, tout = stats["input_tokens"], stats["output_tokens"]
    ext = stats["by_stage"].get("extract", {"tokens": 0})["tokens"]
    share = ext / max(1, tin + tout)
    all_large = cost(tin, tout, "large")
    routed = cost(int(tin * (1 - share)), int(tout * (1 - share)), "large") + cost(int(tin * share), int(tout * share), "small")
    table([{"setup": "large model for every step", "est_usd": f"{all_large:.4f}"},
           {"setup": "large for planning/synthesis, small for extraction", "est_usd": f"{routed:.4f}"},
           {"setup": "small model for every step", "est_usd": f"{cost(tin, tout, 'small'):.4f}"}], max_width=60)
    print("  In code: LLM.small() is used for extraction (set LLM_SMALL_MODEL in .env).")

    step("4. A hard budget: max 10 tool calls")
    tight = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), use_memory=False,
                            budget=Budget(max_tool_calls=10, max_usd=1.0))
    r = tight.run()
    print("  stop reason   :", tight.iterations[-1]["decision"]["reasons"][0])
    print("  fields found  :", [f.field for f in r.key_findings])
    print("  open questions:", *(r.open_questions or ["none"]), sep="\n    • ")
    pause_note("Budgets turn 'search until confident' into 'search until confident OR out of budget' — and the "
               "report honestly lists what is still open instead of guessing. (Parallel workers can overshoot a "
               "little because they check the budget between calls; production systems reserve budget first.)")

    step("5. Latency model (play with the numbers)")
    rows = []
    for workers in (1, 2, 4, 8):
        sub_questions, calls_each, seconds_per_call, llm_overhead = 8, 4, 1.2, 6.0
        waves = -(-sub_questions // workers)  # ceiling division
        rows.append({"parallel workers": workers, "waves": waves,
                     "≈ seconds": round(waves * calls_each * seconds_per_call + llm_overhead, 1)})
    table(rows)
    pause_note("8 sub-questions × 4 tool calls × 1.2 s: one worker takes ~45 s, eight workers ~11 s. "
               "Planning and synthesis stay sequential, so speed-up flattens (Amdahl's law).")


if __name__ == "__main__":
    main()
