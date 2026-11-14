"""
Can we trust the LLM judge? Compare its verdicts with a human reviewer using Cohen's kappa.
By Shivank Agarwal. No model calls: this only reads a CSV.

How to use:
1. Run prompt_evals_inventory.py      -> prompt_eval_results.csv (the judge's scores)
2. Open annotator.html, upload that CSV, mark each answer 👍 / 👎, click "Download feedback"
                                       -> prompt_eval_results_feedback.csv
3. python judge_agreement.py prompt_eval_results_feedback.csv

The judge "passes" an answer only if every score column is 1. The human label is pass / fail.
"""
import csv
import sys

SCORE_COLUMNS = ["accuracy", "clarity", "completeness",                                  # prompt evals
                 "context_relevance", "answer_groundedness", "answer_correctness"]       # RAG evals


def cohens_kappa(a, b):
    """Agreement between two raters, after removing the agreement expected by chance."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    chance = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    kappa = (observed - chance) / (1 - chance) if chance < 1 else 1.0
    return observed, chance, kappa


def describe(k):
    """Common reading of kappa values (Landis & Koch)."""
    for limit, word in [(0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if k <= limit:
            return word
    return "almost perfect"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "prompt_eval_results_feedback.csv"
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_label") in ("pass", "fail")]
    if not rows:
        sys.exit("No rows with a human_label of pass/fail. Annotate the file in annotator.html first.")

    scores = [c for c in SCORE_COLUMNS if c in rows[0]]
    judge = ["pass" if all(r[c] == "1" for c in scores) else "fail" for r in rows]
    human = [r["human_label"] for r in rows]

    print(f"File: {path}  ({len(rows)} reviewed answers)")
    print(f"Judge passes an answer when all of these are 1: {', '.join(scores)}\n")

    print("                 Human: pass   Human: fail")
    for j in ("pass", "fail"):
        cells = [sum(1 for x, y in zip(judge, human) if x == j and y == h) for h in ("pass", "fail")]
        print(f"  Judge: {j:<5}   {cells[0]:>10}   {cells[1]:>11}")

    observed, chance, kappa = cohens_kappa(judge, human)
    print(f"\nRaw agreement        : {observed:.0%}")
    print(f"Agreement by chance  : {chance:.0%}   (judge passes {judge.count('pass')}/{len(rows)}, "
          f"human passes {human.count('pass')}/{len(rows)})")
    print(f"Cohen's kappa        : {kappa:.2f}  → {describe(kappa)} agreement")

    disagreements = [(r, j, h) for r, j, h in zip(rows, judge, human) if j != h]
    if disagreements:
        print(f"\nDisagreements to discuss ({len(disagreements)}):")
        for r, j, h in disagreements:
            tag = r.get("prompt_version") or r.get("config") or ""
            print(f"  • [{tag}] {r['question'][:70]}...")
            print(f"      judge: {j:<4} ({r.get('judge_reason', '')})")
            print(f"      human: {h:<4} ({r.get('human_feedback') or 'no note'})")
