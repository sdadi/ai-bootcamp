"""
Chapter 1.2 · Leaked question papers: spotting benchmark contamination
AI Evals in Practice · Author: Shivank Agrawal

If a famous test question was in the model's training data, a high score proves memory, not skill.
A quick check: give the model the FIRST half of the item and ask it to continue the original text.
If it reproduces the rest word for word, the item has probably leaked.

Run:  python ch1_02_contamination_check.py
"""
from openai import OpenAI

client = OpenAI()

# Famous public puzzles: (first half, the real second half)
ITEMS = [
    ("A farmer has 17 sheep. All but 9 run away. How many",
     "sheep does the farmer have left?"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much",
     "does the ball cost?"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long",
     "would it take 100 machines to make 100 widgets?"),
]


def overlap(expected, produced):
    """Share of the expected words that appear, in order, at the start of what the model produced."""
    exp = expected.lower().split()
    got = produced.lower().replace("\n", " ").split()
    same = 0
    for e, g in zip(exp, got):
        if e.strip("?.,") != g.strip("?.,"):
            break
        same += 1
    return same / len(exp)


for half, truth in ITEMS:
    r = client.chat.completions.create(
        model="gpt-6.1-terra", reasoning_effort="none", temperature=0,
        messages=[{"role": "user", "content":
                   "Continue this text exactly as it appears in its original source. "
                   "Write only the continuation:\n" + half}])
    produced = r.choices[0].message.content.strip()
    score = overlap(truth, produced)
    flag = "LIKELY LEAKED" if score >= 0.8 else "not reproduced"
    print(f"{score:4.0%}  {flag:15} | ...{' '.join(half.split()[-5:])} -> {produced[:60]!r}")

# Example output:
# 100%  LIKELY LEAKED   | ...9 run away. How many -> 'sheep does the farmer have left? Answer: 9'
# 100%  LIKELY LEAKED   | ...than the ball. How much -> 'does the ball cost?'
# 100%  LIKELY LEAKED   | ...make 5 widgets, how long -> 'would it take 100 machines to make 100 widgets?'
#
# Lesson: public benchmark items are not a fair test of YOUR product. Build your own test set (Chapter 3).
