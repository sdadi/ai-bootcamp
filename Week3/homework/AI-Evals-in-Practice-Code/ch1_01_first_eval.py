"""
Chapter 1.1 · Your first eval in code
AI Evals in Practice · Author: Shivank Agrawal

An eval is just an exam for your AI:
  test set      = the question paper  (here: 3 subjects x 2 questions)
  marking rule  = how each answer is checked (here: a keyword a correct answer must contain)
  score         = pass rate per subject

Run:  python ch1_01_first_eval.py
"""
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from the environment

SYSTEM = "You are DoubtBot, a DSA teaching assistant. Explain briefly. Give hints, not full solutions."

# Each test set = one "subject". Each test case = question + a word a correct answer must contain.
TEST_SETS = {
    "recursion": [
        {"q": "Why is recursive fibonacci(50) so slow?",       "must": "memo"},
        {"q": "What is a base case?",                          "must": "stop"}],
    "binary_search": [
        {"q": "Binary search needs the array to be what?",     "must": "sorted"},
        {"q": "Time complexity of binary search?",             "must": "log"}],
    "heaps": [
        {"q": "In a min-heap, where is the smallest element?", "must": "root"},
        {"q": "Cost of pushing into a heap of n items?",       "must": "log"}],
}


def ask_doubtbot(question):
    r = client.chat.completions.create(
        model="gpt-6.1-terra",
        reasoning_effort="none", temperature=0,      # temperature is only allowed with reasoning off
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}])
    return r.choices[0].message.content


for subject, cases in TEST_SETS.items():
    passed = 0
    for case in cases:
        answer = ask_doubtbot(case["q"])
        ok = case["must"] in answer.lower()          # the "marking scheme"
        passed += ok
        if not ok:                                   # always look at the failures, not just the score
            print(f"   ✗ {case['q']!r}: no '{case['must']}' in -> {answer[:80]!r}...")
    print(f"{subject:14} {passed}/{len(cases)} = {passed / len(cases):.0%}")

# Example output (yours may differ):
# recursion      2/2 = 100%
# binary_search  2/2 = 100%
#    ✗ 'Cost of pushing into a heap of n items?': no 'log' in -> 'O(lg n): the new item sifts up...'
# heaps          1/2 = 50%
#
# Look at the failure: the answer was RIGHT ("lg n") but the keyword rule said no.
# Keyword checks are cheap but brittle. Chapter 3 replaces them with an LLM judge.
