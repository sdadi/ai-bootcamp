"""
Chapter 2.3 · Model-based evals: comparing an answer with a reference
AI Evals in Practice · Author: Shivank Agrawal

Two classic ways to compare a bot answer with the TA's reference answer:
  1. Token overlap (ROUGE-1 recall): how many reference words also appear in the answer?
  2. Semantic similarity: cosine similarity of OpenAI embeddings (does it MEAN the same?)

Watch the second answer: it says the OPPOSITE ("unsorted") yet scores high on both.
Neither metric checks logic. That is why Chapter 3 uses an LLM judge with a rubric.

Run:  python ch2_02_reference_metrics.py
"""
import re
from collections import Counter

import numpy as np
from openai import OpenAI

client = OpenAI()


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge1_recall(reference, answer):
    """Share of reference words (with counts) that the answer also contains."""
    ref, ans = Counter(words(reference)), Counter(words(answer))
    overlap = sum(min(count, ans[w]) for w, count in ref.items())
    return overlap / sum(ref.values())


def embed(texts):
    out = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in out.data])          # vectors come normalised to length 1


reference = "binary search needs a sorted array"
answers = ["it only works on ordered lists",                   # right, different words
           "binary search needs an unsorted array",            # WRONG, almost the same words
           "heaps keep the smallest item at the root"]         # off-topic

vecs = embed([reference] + answers)
print("ROUGE-1  cosine   answer")
for a, v in zip(answers, vecs[1:]):
    print(f"  {rouge1_recall(reference, a):.2f}    {vecs[0] @ v:.2f}   {a}")   # dot product = cosine here

# Example output (cosine values vary a little by model version):
# ROUGE-1  cosine   answer
#   0.00    0.71   it only works on ordered lists
#   0.67    0.88   binary search needs an unsorted array
#   0.00    0.21   heaps keep the smallest item at the root
#
# The correct paraphrase gets ROUGE 0.00; the wrong answer wins on BOTH metrics.
