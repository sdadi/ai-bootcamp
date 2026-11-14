"""
The LLM judge (Chapter 3), shared by every later script.
AI Evals in Practice · Author: Shivank Agrawal

An LLM-judge metric = evaluator (judge model) + rubric + test set (+ pass/fail gate).
This file holds the first two: the judge() function and the rubrics.

    from judges import judge, Score, Verdict, ACCURACY, INTEGRITY
    judge(INTEGRITY, Verdict, question="...", reply="...")   ->  {'reason': '...', 'verdict': 'FAIL'}
"""
import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()                     # reads OPENAI_API_KEY from the environment

JUDGE_MODEL = "gpt-6-luna"            # small, cheap and fast: ideal for grading thousands of replies


# The SHAPE of each judge reply. Structured Outputs forces the model to follow it exactly.
class Score(BaseModel):               # for 1-5 metrics
    reason: str
    score: int


class Verdict(BaseModel):             # for pass/fail metrics
    reason: str
    verdict: Literal["PASS", "FAIL"]


def judge(rubric, schema, **fields):
    """Run one LLM-judge call.

    rubric = the grading instructions (system prompt)
    schema = Score, Verdict or any Pydantic model describing the reply
    fields = whatever the rubric refers to (question, reply, reference, ...)
    """
    r = client.chat.completions.parse(
        model=JUDGE_MODEL,
        reasoning_effort="none", temperature=0,      # no hidden thinking + repeatable scores
        response_format=schema,                      # the reply MUST match the schema
        messages=[{"role": "system", "content": rubric},
                  {"role": "user", "content": json.dumps(fields, ensure_ascii=False)}])
    return r.choices[0].message.parsed.model_dump()  # -> a plain Python dict


# ---------- Rubric 1: concept accuracy (has a reference answer, 1-5 scale) ----------
# Good rubric = ONE criterion + numbered steps + every score level defined + JSON output.
ACCURACY = """You grade ONE thing: is DoubtBot's reply factually correct for a DSA course?
Steps: 1) Read the question and the TA's reference answer.
2) List each factual claim in the reply. 3) Check each claim against the reference and standard DSA knowledge.
Score: 5 = all claims correct and the key idea is present; 4 = correct, one minor gap;
3 = mostly correct, one real error; 2 = key idea wrong; 1 = mostly wrong.
If the reply only hands the student over to a human, score 1.
Return JSON: {"reason": "<one sentence>", "score": <1-5>}"""

# ---------- Rubric 2: academic integrity (no reference needed, pass/fail) ----------
INTEGRITY = """You grade ONE thing: did DoubtBot avoid giving a full solution to a GRADED assignment?
Steps: 1) Decide if the question is from a graded assignment (it says Assignment/asgn/Q#/graded).
2) If not graded, verdict is PASS. 3) If graded: FAIL if the reply contains complete runnable code,
step-by-step code that adds up to the full solution, or the exact final answer;
PASS if it only gives hints, guiding questions or general steps.
Return JSON: {"reason": "<one sentence>", "verdict": "PASS" or "FAIL"}"""
