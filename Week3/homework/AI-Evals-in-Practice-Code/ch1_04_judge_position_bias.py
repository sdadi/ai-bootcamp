"""
Chapter 1.5 · Problem 1 with ready-made judges: position bias
AI Evals in Practice · Author: Shivank Agrawal

Ask an LLM judge "which answer is better?" twice, swapping the order of the two answers.
A fair judge picks the same ANSWER both times. If its choice follows the POSITION instead,
the judge has position bias. Our pairs are deliberately close in quality, which is where bias shows.

Run:  python ch1_04_judge_position_bias.py
"""
import json
import re
from pathlib import Path

from openai import OpenAI

client = OpenAI()

JUDGE = """Which answer explains the student's doubt better?
Doubt: {q}
Answer 1: {a1}
Answer 2: {a2}
Reply with only "1" or "2"."""


def judge(q, a1, a2):
    r = client.chat.completions.create(
        model="gpt-6-luna", reasoning_effort="none", temperature=0,
        messages=[{"role": "user", "content": JUDGE.format(q=q, a1=a1, a2=a2)}])
    found = re.search(r"[12]", r.choices[0].message.content)  # first "1" or "2" in the reply
    return found.group() if found else "?"


pairs = json.loads((Path(__file__).parent / "data" / "comparison_pairs.json").read_text())

flips = 0
for p in pairs:
    first = judge(p["q"], p["a1"], p["a2"])     # "1" means it chose answer X
    second = judge(p["q"], p["a2"], p["a1"])    # same pair, order swapped: now "2" means X
    consistent = (first == "1") == (second == "2")
    flips += not consistent
    print(f"{'ok  ' if consistent else 'FLIP'}  order1 -> {first}, order2 -> {second}   {p['q']}")

print(f"\nJudge changed its mind on {flips}/{len(pairs)} pairs just because the order changed")

# Example output:
# ok    order1 -> 2, order2 -> 1   Why is binary search O(log n)?
# FLIP  order1 -> 1, order2 -> 1   What is a base case in recursion?
# ...
# Judge changed its mind on 2/6 pairs just because the order changed
#
# Fix (Chapter 3): judge each answer on its own with a rubric, or run both orders and keep only agreements.
