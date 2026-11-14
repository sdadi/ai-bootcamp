"""
Chapter 1.3 · An AI product is not the same as an AI model
AI Evals in Practice · Author: Shivank Agrawal

Same model, two system prompts = two different PRODUCTS with different behaviour.
We count how often each one hands out a full code solution to a graded assignment.

Run:  python ch1_03_product_vs_model.py
"""
import json
from pathlib import Path

from openai import OpenAI

client = OpenAI()

PROMPTS = {
    "A": "You are a helpful DSA tutor.",
    "B": "You are a helpful DSA tutor. Never give full code solutions. Give hints only.",
}

# 10 student doubts, each tagged "concept" or "assignment"
doubts = json.loads((Path(__file__).parent / "data" / "test_doubts.json").read_text())
assignments = [d for d in doubts if d["type"] == "assignment"]

for name, system in PROMPTS.items():
    full_solutions = 0
    for d in assignments:
        answer = client.chat.completions.create(
            model="gpt-6.1-terra", reasoning_effort="none", temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": d["text"]}]).choices[0].message.content
        if "def " in answer:            # crude check: did it write a full Python function?
            full_solutions += 1
    print(f"Prompt {name}: full solutions given on {full_solutions}/{len(assignments)} graded assignments")

# Example output:
# Prompt A: full solutions given on 5/5 graded assignments
# Prompt B: full solutions given on 0/5 graded assignments
#
# The model did not change; the product did. So evals must test the PRODUCT
# (prompt + retrieval + tools + model), not just the model's leaderboard score.
