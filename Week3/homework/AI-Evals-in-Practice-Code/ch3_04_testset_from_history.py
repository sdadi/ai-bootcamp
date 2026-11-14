"""
Chapter 3.9 (II) · Test cases from historical data
AI Evals in Practice · Author: Shivank Agrawal

Old forum threads are a goldmine: real student wording + a TA's verified answer.
An LLM turns each messy thread into one clean test case. Two rules in the prompt matter:
  - keep the student's intent when rewriting the question
  - use ONLY the TA's reply for the expected answer (ignore other students' guesses)
The result is a "silver" test set; after a TA reviews it, it becomes "golden".

Run:  python ch3_04_testset_from_history.py
"""
import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

EXTRACT = """Turn this course-forum thread into ONE evaluation test case.
Rewrite the student's question so it is clear and self-contained (keep the student's intent).
Use ONLY the TA's reply as the basis for the expected answer; ignore other students' guesses.
topic: one short label such as recursion, graphs/dfs, heaps, sorting, searching, admin."""


class TestCase(BaseModel):
    question: str
    expected_answer: str
    topic: str


HERE = Path(__file__).parent
threads = json.loads((HERE / "data" / "forum_threads.json").read_text())
(HERE / "outputs").mkdir(exist_ok=True)

silver = []
for t in threads:
    thread = {"post": t["post"], "replies": t["replies"]}           # what the LLM sees
    r = client.chat.completions.parse(
        model="gpt-6.1-terra", reasoning_effort="none", temperature=0,
        response_format=TestCase,
        messages=[{"role": "system", "content": EXTRACT},
                  {"role": "user", "content": json.dumps(thread, ensure_ascii=False)}])
    case = r.choices[0].message.parsed.model_dump()
    case["source_thread"] = t["id"]                                  # keep the link for reviewers
    silver.append(case)

print(json.dumps(silver[0], indent=1, ensure_ascii=False))
(HERE / "outputs" / "silver_testset.json").write_text(json.dumps(silver, indent=1, ensure_ascii=False))
print(f"\n{len(silver)} silver test cases saved to outputs/silver_testset.json (send to a TA for review)")

# Example output:
# {
#  "question": "My recursive DFS on a grid works on small inputs but never finishes on larger ones. Why?",
#  "expected_answer": "The DFS does not mark cells as visited, so it revisits neighbours endlessly. Keep a visited set and check it before each recursive call.",
#  "topic": "graphs/dfs",
#  "source_thread": "T101"
# }
#
# 6 silver test cases saved to outputs/silver_testset.json (send to a TA for review)
