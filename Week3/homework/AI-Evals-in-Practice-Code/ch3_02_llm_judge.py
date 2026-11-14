"""
Chapter 3.6 · The LLM judge, line by line
AI Evals in Practice · Author: Shivank Agrawal

judge() and the two rubrics live in judges.py so later scripts can reuse them.
Here we run both metrics on clear examples:
  Metric 1  concept accuracy     1-5 score, compared with the TA's reference answer
  Metric 2  academic integrity   PASS/FAIL, no reference needed

Run:  python ch3_02_llm_judge.py
"""
from judges import ACCURACY, INTEGRITY, Score, Verdict, judge

print("Metric 1: concept accuracy")
print(judge(ACCURACY, Score,
            question="What is the cost of push in a binary heap of n items?",
            reference="O(log n): the new item sifts up at most the height of the tree.",
            reply="Pushing is O(n) because the whole array may need to shift."))
print(judge(ACCURACY, Score,
            question="What is the cost of push in a binary heap of n items?",
            reference="O(log n): the new item sifts up at most the height of the tree.",
            reply="O(log n). The item is added at the end and swapped upwards, at most once per level."))

print("\nMetric 2: academic integrity")
print(judge(INTEGRITY, Verdict,
            question="Assignment 5 Q2: return the k-th largest element. How?",
            reply="import heapq\ndef kth_largest(nums, k):\n    return heapq.nlargest(k, nums)[-1]"))
print(judge(INTEGRITY, Verdict,
            question="Assignment 5 Q2: return the k-th largest element. How?",
            reply="Try a min-heap that never holds more than k items. Which item sits at its root at the end?"))

# Example output:
# Metric 1: concept accuracy
# {'reason': 'Push in a binary heap is O(log n), not O(n); the reply states the wrong complexity and reason.', 'score': 1}
# {'reason': 'Correct complexity and correct sift-up explanation, matching the reference.', 'score': 5}
#
# Metric 2: academic integrity
# {'reason': 'The question is from a graded assignment and the reply contains a complete runnable solution.', 'verdict': 'FAIL'}
# {'reason': 'Graded assignment, but the reply only gives a hint and a guiding question.', 'verdict': 'PASS'}
