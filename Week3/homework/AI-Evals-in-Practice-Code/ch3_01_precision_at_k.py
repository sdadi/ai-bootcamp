"""
Chapter 3.3 · A component metric: retrieval precision@k
AI Evals in Practice · Author: Shivank Agrawal

End-to-end metrics tell you THAT DoubtBot failed; component metrics tell you WHERE.
precision@k = of the top k notes retrieved, how many are actually relevant?
TAs labelled which notes are relevant for each question (data/retrieval_labels.json).

Run:  python ch3_01_precision_at_k.py
"""
import json
from pathlib import Path

from doubtbot import retrieve          # only the retrieval step of our product


def precision_at_k(retrieved_ids, relevant_ids, k=2):
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc in top_k if doc in relevant_ids)
    return hits / k


# 1) The worked example from the notes, by hand
relevant = {"L14_heaps", "L15_heapify"}
retrieved = ["L15_heapify", "wiki_jvm_memory", "L14_heaps", "L09_trees"]
print("worked example, precision@3 =", round(precision_at_k(retrieved, relevant, k=3), 2))

# 2) The real retriever on labelled questions
labels = json.loads((Path(__file__).parent / "data" / "retrieval_labels.json").read_text())
scores = []
for item in labels:
    got = [c["id"] for c in retrieve(item["question"], k=2)]
    p = precision_at_k(got, set(item["relevant"]), k=2)
    scores.append(p)
    print(f"P@2={p:.1f}  got={got}  {item['question']}")
print(f"\nmean precision@2 = {sum(scores) / len(scores):.2f}")

# Example output:
# worked example, precision@3 = 0.67
# P@2=1.0  got=['L15_heapify', 'L14_heaps']  Why is building a heap O(n) and not O(n log n)?
# P@2=0.5  got=['L06_binary_search', 'L03_sorting']  Why does my binary search give wrong answers on [5,1,9,3]?
# P@2=1.0  got=['L11_recursion', 'L17_dp']  My recursive fibonacci is super slow for n=45, why?
# P@2=0.5  got=['L12_dfs', 'L09_trees']  DFS on a grid never stops, what am I missing?
# P@2=0.5  got=['L03_sorting', 'L06_binary_search']  What is the time complexity of Python's sorted()?
#
# mean precision@2 = 0.70
# Note: when a question has only 1 relevant note, the best possible precision@2 is 0.5.
