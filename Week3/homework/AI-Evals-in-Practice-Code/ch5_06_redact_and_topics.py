"""
Chapter 5.7 · Clean data before you analyse it: redact, then classify
AI Evals in Practice · Author: Shivank Agrawal

Before analysts look at chat logs:
  1. REDACT personal data with rules (phone, email, student ID), before any model sees it
  2. CLASSIFY each first message into a fixed topic list with a cheap model
  3. analysts see only the COUNTS, never the raw messages

Structured Outputs with a Literal means the topic can ONLY be one of the allowed labels.
"sensitive" catches personal or emotional disclosures so they are routed away from analytics.

Run:  python ch5_06_redact_and_topics.py
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

REDACT = [(re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
          (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
          (re.compile(r"\bDSA\d{2}-\d{4}\b"), "[STUDENT_ID]")]


def redact(text):
    for pattern, tag in REDACT:
        text = pattern.sub(tag, text)
    return text


class Topic(BaseModel):
    topic: Literal["arrays", "recursion", "sorting", "searching", "heaps", "graphs", "dp",
                   "admin", "other", "sensitive"]


TOPIC = """Classify this student doubt into one topic:
arrays, recursion, sorting, searching, heaps, graphs, dp, admin, other.
If it contains a personal or emotional disclosure, use topic "sensitive" and nothing else."""


def topic(text):
    r = client.chat.completions.parse(
        model="gpt-6-luna", reasoning_effort="none", temperature=0,
        response_format=Topic,                       # the topic can ONLY be one of the allowed labels
        messages=[{"role": "system", "content": TOPIC},
                  {"role": "user", "content": redact(text)}])   # the model never sees raw PII
    return r.choices[0].message.parsed.topic


messages = json.loads((Path(__file__).parent / "data" / "first_messages.json").read_text())

print("Redaction examples:")
for m in messages:
    if redact(m) != m:
        print("  ", redact(m))

counts = Counter(topic(m) for m in messages)          # analysts only ever see these counts
print("\nThis week's topics:", counts.most_common())

# Example output:
# Redaction examples:
#    bfs shortest path not working on my graph, my roll is [STUDENT_ID]
#    fees EMI deducted twice, call me on [PHONE]
#    how to change my batch timing? mail me at [EMAIL]
#
# This week's topics: [('graphs', 3), ('heaps', 2), ('recursion', 2), ('dp', 2), ('admin', 2), ('sorting', 1),
#                      ('searching', 1), ('arrays', 1), ('sensitive', 1), ('other', 1)]
