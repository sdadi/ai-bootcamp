"""
Chapter 2.6 · Building an error taxonomy from open-coding notes
AI Evals in Practice · Author: Shivank Agrawal

Open coding: TAs read traces and write a free-text note for each failure (data/open_coding_notes.txt).
Axial coding: group those notes into a small set of failure categories that are
MECE (mutually exclusive, collectively exhaustive). An LLM does the first draft; humans review it.

Structured Outputs (client.chat.completions.parse + a Pydantic model) makes the reply
arrive as a ready Python object, so there is no json.loads and no broken JSON.

Run:  python ch2_04_error_taxonomy.py
"""
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

notes = (Path(__file__).parent / "data" / "open_coding_notes.txt").read_text().strip().splitlines()

prompt = """You are helping build an error taxonomy for an AI teaching assistant.
Group these reviewer notes into 5-9 failure categories that are mutually exclusive
and together cover every note. Name each category in 2-4 words.
List the note numbers that belong to each category; every note goes in exactly one category.

Notes:
""" + "\n".join(f"{i}. {n}" for i, n in enumerate(notes))


class Category(BaseModel):          # the shape of ONE category
    name: str
    definition: str
    note_ids: list[int]


class Taxonomy(BaseModel):          # the shape of the whole reply
    categories: list[Category]


r = client.chat.completions.parse(
    model="gpt-6.1-terra", reasoning_effort="none", temperature=0,
    response_format=Taxonomy,                         # the reply MUST match Taxonomy
    messages=[{"role": "user", "content": prompt}])

taxonomy = r.choices[0].message.parsed                # already a Python object
for c in sorted(taxonomy.categories, key=lambda c: -len(c.note_ids)):
    print(f"{c.name:22} {len(c.note_ids):3} notes  - {c.definition}")

# A quick MECE check you should always run on an LLM's grouping:
used = [i for c in taxonomy.categories for i in c.note_ids]
missing = sorted(set(range(len(notes))) - set(used))
doubled = sorted({i for i in used if used.count(i) > 1})
print(f"\n{len(notes)} notes | not covered: {missing or 'none'} | in two categories: {doubled or 'none'}")

# Example output:
# Direct solution          4 notes  - Gives full code or the final answer for graded work
# Retrieval failure        4 notes  - Fetched notes are irrelevant or from the wrong source
# Level mismatch           4 notes  - Explanation far above the student's level
# Missed escalation        4 notes  - Should have handed over to a human but did not
# Over-escalation          3 notes  - Handed over even though the notes answer it
# Factual error            3 notes  - States something wrong about DSA
# Cold or curt tone        3 notes  - Correct but discouraging or robotic
# Slow response            3 notes  - Took more than 10 seconds
#
# 28 notes | not covered: none | in two categories: none
