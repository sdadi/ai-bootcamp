"""
LAB 13 - Tracing, latency percentiles and cost  (Module 12)
==========================================================
What you will learn
  * One trace per request, one span per stage: you can see WHERE time goes.
  * Report p50 / p95 / p99, not averages: users feel the slow tail.
  * Estimate monthly cost before you launch, and see which knob saves the most.

How to run
    python labs/lab13_observability_and_cost.py

Output
    logs/traces.jsonl   (one JSON line per question; open it or load it into pandas)
"""
import json

import _common
from rag_core.evaluation import load_golden
from rag_core.observability import estimate_cost, percentiles
from rag_core.pipeline import RAGPipeline
from rag_core.utils import banner, table

embedder = _common.embedder_or_exit()
llm = _common.llm_status()
p = RAGPipeline(embedder=embedder, llm=llm).build()
log = p.tracer.sink
start = sum(1 for _ in open(log, encoding="utf-8")) if log.exists() else 0

banner("1. Run 30 questions with tracing on")
for item in load_golden()[:30]:
    p.ask(item["question"], _common.default_groups(item), item.get("include_archived", False))
traces = [json.loads(l) for l in open(log, encoding="utf-8").read().splitlines()[start:]]
print(f"{len(traces)} traces written to {log}")
print("Example trace:\n", json.dumps(traces[0], indent=1, ensure_ascii=False)[:900])

banner("2. Latency per stage (ms)")
rows = []
for stage in ["retrieve", "rerank", "generate"]:
    vals = [s["ms"] for t in traces for s in t["spans"] if s["name"] == stage]
    if vals:
        rows.append({"stage": stage, **percentiles(vals)})
rows.append({"stage": "TOTAL", **percentiles([t["total_ms"] for t in traces])})
table(rows)
print("""Realistic production numbers (cloud LLM, warm services):
  query embedding 5-50 ms | vector + keyword search 5-50 ms | reranking 25-50 chunks 50-300 ms
  LLM time-to-first-token 300 ms - 2 s | full answer 1-10 s depending on length and model.
So: STREAM the answer (users see text after the first token), and budget the stages before the LLM.""")

banner("3. What will it cost per month?")
queries_per_day = 10_000
prompt_tokens = 300 + 5 * 250        # system prompt + 5 chunks of ~250 tokens
output_tokens = 150
prices = {  # USD per 1M tokens (input, output). Check your provider's pricing page; these change often.
    "small/cheap model (e.g. gpt-6-luna)": (0.10, 0.50),
    "mid-size model": (1.00, 5.00),
    "frontier model": (5.00, 25.00),
}
rows = []
for name, (pin, pout) in prices.items():
    per_q = estimate_cost(prompt_tokens, output_tokens, pin, pout)
    rows.append({"model tier": name, "USD / query": f"{per_q:.5f}", "USD / month": f"{per_q * queries_per_day * 30:,.0f}"})
table(rows)
print(f"""Assumptions: {queries_per_day:,} questions/day, {prompt_tokens} input + {output_tokens} output tokens each.
Biggest savings, in order: fewer/smaller chunks in the prompt (reranking helps you keep only 3-5),
a cheaper model for easy questions (routing), semantic caching for repeated questions,
prompt caching of the fixed system prompt, and batch APIs for offline work like ingestion.
Embedding 1M chunks of 250 tokens with a $0.02/1M-token model costs about $5 - ingestion is rarely the big bill.""")
