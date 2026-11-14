"""
Chapter 2.7 · No traces yet? Trace bootstrapping
AI Evals in Practice · Author: Shivank Agrawal

Before launch you have no real user traffic to review. But you DO have history:
old forum threads where a human TA already answered. Replay those questions through
the product, record every step (a "trace"), and keep the TA's answer as ground truth.

Then open outputs/traces.jsonl and read it like a TA would: that is open coding (Chapter 2.6).

Run:  python ch2_03_bootstrap_traces.py
"""
import json
from pathlib import Path

from doubtbot import answer          # our product

HERE = Path(__file__).parent
threads = json.loads((HERE / "data" / "forum_threads.json").read_text())   # 6 sampled doubt threads
(HERE / "outputs").mkdir(exist_ok=True)

with open(HERE / "outputs" / "traces.jsonl", "w") as out:
    for t in threads:
        trace = answer(t["post"]["text"])                       # run the product, with its steps recorded
        out.write(json.dumps({
            "thread_id":  t["id"],
            "question":   t["post"]["text"],
            "ta_answer":  t["accepted_ta_reply"],                 # ground truth: what the human TA did
            "retrieved":  [(c["id"], c["score"]) for c in trace["retrieved"]],   # which notes were fetched
            "bot_answer": trace["answer"],
            "escalated":  trace["escalated"],
            "latency_s":  trace["latency_s"],
        }, ensure_ascii=False) + "\n")
        print(f"{t['id']}  escalated={trace['escalated']!s:5}  top note={trace['retrieved'][0]['id']:18}  {t['post']['text'][:50]}")

print(f"\nWrote {len(threads)} traces to outputs/traces.jsonl. Read them and write one note per failure.")

# Example output:
# T101  escalated=False  top note=L12_dfs             why does my dfs on the grid never end?? it works o
# T102  escalated=False  top note=L15_heapify         heapify is O(n)?? sir said heappush is log n so n
# T103  escalated=False  top note=L11_recursion       fib(45) recursive takes forever on my laptop. is m
# T104  escalated=False  top note=L06_binary_search   binary search returning wrong index sometimes, arr
# T105  escalated=True   top note=ADMIN_helpline      my fees EMI got deducted twice this month, who do
# T106  escalated=False  top note=L03_sorting         quick sort took ages on an already sorted list of
#
# Wrote 6 traces to outputs/traces.jsonl. Read them and write one note per failure.
# (T105 is a finding already: the helpline note answers it, so escalating was not needed.)
