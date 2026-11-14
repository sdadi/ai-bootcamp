"""
Chapter 3.7 · Checking the judge against humans (calibration)
AI Evals in Practice · Author: Shivank Agrawal

Before you trust a judge, compare it with expert labels on the same cases.
TAs labelled 16 replies PASS/FAIL for academic integrity (data/calibration_set.json).
We treat FAIL as the "positive" class, because catching violations is the judge's job.

  agree  share of cases where judge and TAs give the same label
  TPR    of the real violations, how many did the judge catch?    (missed cheating hurts)
  TNR    of the fine replies, how many did the judge correctly pass? (false alarms annoy)
  kappa  agreement corrected for luck: 0 = chance level, 1 = perfect (aim for 0.6+)

Run:  python ch3_03_judge_calibration.py
"""
import json
from pathlib import Path

from judges import INTEGRITY, Verdict, judge


def agreement_report(human, judge_labels, positive="FAIL"):
    pairs = list(zip(human, judge_labels))
    tp = sum(h == positive and j == positive for h, j in pairs)    # both say violation
    fn = sum(h == positive and j != positive for h, j in pairs)    # judge missed a violation
    fp = sum(h != positive and j == positive for h, j in pairs)    # judge raised a false alarm
    tn = len(pairs) - tp - fn - fp                                 # both say fine
    n = len(pairs)
    agree = (tp + tn) / n
    p_chance = ((tp + fn) / n) * ((tp + fp) / n) + ((fp + tn) / n) * ((fn + tn) / n)
    kappa = (agree - p_chance) / (1 - p_chance)
    return dict(agree=round(agree, 2),
                tpr=round(tp / (tp + fn), 2) if tp + fn else None,
                tnr=round(tn / (tn + fp), 2) if tn + fp else None,
                kappa=round(kappa, 2))


cases = json.loads((Path(__file__).parent / "data" / "calibration_set.json").read_text())

human_labels, judge_labels = [], []
for c in cases:
    j = judge(INTEGRITY, Verdict, question=c["question"], reply=c["reply"])["verdict"]
    human_labels.append(c["human_verdict"])
    judge_labels.append(j)
    if j != c["human_verdict"]:                       # disagreements are where you improve the rubric
        print(f"disagree on {c['id']}: TA={c['human_verdict']} judge={j} | {c['question'][:55]}")

print(agreement_report(human_labels, judge_labels))

# Self-check of the formula with fixed labels (no API): 3 TP, 1 FN, 1 FP, 5 TN
print("formula check:", agreement_report(["FAIL"] * 4 + ["PASS"] * 6,
                                         ["FAIL", "FAIL", "FAIL", "PASS", "FAIL"] + ["PASS"] * 5))

# Example output:
# disagree on C10: TA=FAIL judge=PASS | Assignment 3 Q4: reverse a linked list in place.
# {'agree': 0.94, 'tpr': 0.83, 'tnr': 1.0, 'kappa': 0.86}
# formula check: {'agree': 0.8, 'tpr': 0.75, 'tnr': 0.83, 'kappa': 0.58}
#
# Read each disagreement, sharpen the rubric (e.g. "step-by-step code that adds up to the
# full solution is a FAIL"), and re-run until TPR, TNR and kappa clear your bar.
