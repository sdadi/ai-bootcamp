"""
Prompt evals: is prompt Version 2 really better than Version 1?
By Shivank Agarwal. Model: OpenAI gpt-6-luna (both the advisor and the judge).

1. A golden dataset: 9 tricky inventory questions, each with the key points a correct answer must make.
2. Two system prompts for the same advisor.
3. An LLM judge scores every answer Pass (1) or Fail (0) on Accuracy, Clarity and Completeness,
   and gives a one-line reason, which is what you read during error analysis.
4. Pass rates per prompt version, and a CSV of every answer for human review.
"""
import csv
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                                        # reads OPENAI_API_KEY from .env
client = OpenAI()
MODEL = "gpt-6-luna"
METRICS = ["Accuracy", "Clarity", "Completeness"]
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_eval_results.csv")


def evaluate_response(question, expected_answer, actual_response):
    """LLM-as-judge: strict binary scores (0 = Fail, 1 = Pass) plus a one-line reason."""
    eval_prompt = f"""Evaluate this inventory reorder advisor response with STRICT binary scoring (0=Fail, 1=Pass).

    Question: {question}
    Expected Key Points: {expected_answer}
    Actual Response: {actual_response}

    Be STRICT. Only give 1 if the criterion is fully met:
    - Accuracy: Is the reorder recommendation correct with NO errors or misleading statements? (0 or 1)
    - Clarity: Would a warehouse manager act on this WITHOUT further questions? (0 or 1)
    - Completeness: Does it address the SPECIFIC edge case or nuance in the question? (0 or 1)

    Respond ONLY with JSON in this exact shape:
    {{"Accuracy": 0 or 1, "Clarity": 0 or 1, "Completeness": 0 or 1, "Reason": "one short sentence"}}"""

    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        response_format={"type": "json_object"},     # JSON mode: the reply is always valid JSON
        messages=[{"role": "user", "content": eval_prompt}],
    )
    verdict = json.loads(response.choices[0].message.content)
    scores = {m: 1 if verdict.get(m) in (1, "1", True) else 0 for m in METRICS}
    scores["Reason"] = str(verdict.get("Reason", ""))
    return scores


def inventory_advisor(system_prompt, question):
    """The agent under test: answers an inventory question using the given system prompt."""
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


dataset = [
    {
        "q": "Our safety stock formula says reorder at 200 units. We have 210 units on hand and a PO arriving tomorrow. Should we reorder now?",
        "a": "No. On-hand (210) is above reorder point (200) AND incoming stock will arrive tomorrow. Reordering now creates overstock. Must account for in-transit inventory in the reorder calculation.",
    },
    {
        "q": "Demand jumped 40% last week. Should I immediately raise the reorder point by 40%?",
        "a": "No. A single week's spike may be seasonal or anomalous. Must consider average demand over a rolling window (e.g., 4–8 weeks). Raising the reorder point by 40% based on one week risks permanent overstock.",
    },
    {
        "q": "We use EOQ to calculate order quantity. Our carrying cost just doubled. Should my order quantity go up or down?",
        "a": "Down. EOQ = sqrt(2DS/H). As H (holding cost) doubles, EOQ decreases. Ordering in smaller, more frequent batches is now cheaper than holding large lots.",
    },
    {
        "q": "Lead time from our supplier is 7 days. My reorder point is 70 units (10 units/day × 7 days). Is that correct?",
        "a": "Only in the ideal case with zero demand variability. In practice the reorder point must include safety stock: ROP = (avg demand × lead time) + safety stock. The formula is incomplete without safety stock for variable demand.",
    },
    {
        "q": "We have a perishable item with a 10-day shelf life. Our reorder cycle is every 8 days. Is there a problem?",
        "a": "Potentially yes. If the item arrives on day 1 and sits until the next order cycle, remaining shelf life when consumed could be very short. Must align reorder cycle with shelf life and consider FEFO (First Expired First Out) rotation.",
    },
    {
        "q": "Our ABC analysis puts item X in class C (low value). Can I just set a fixed reorder point and ignore it?",
        "a": "Not always. Class C items can still be operationally critical (e.g., a cheap bolt that stops an assembly line). Must distinguish between value-based ABC and criticality. Critical C items need tighter review despite low cost.",
    },
    {
        "q": "Supplier A has a unit price of $10 with 30-day lead time. Supplier B has $11 with 5-day lead time. Supplier A is always better, right?",
        "a": "No. Longer lead time requires higher safety stock, which increases holding costs. Total cost = purchase price + ordering cost + holding cost. Supplier B's shorter lead time may result in lower total inventory cost despite higher unit price.",
    },
    {
        "q": "We ran out of stock last month. I want to set safety stock high enough that we never stock out again. Is that achievable?",
        "a": "Not economically. Safety stock reduces stockout probability but cannot eliminate it without infinite inventory. Must choose a service level (e.g., 95% or 99%) and calculate the safety stock that achieves it, accepting a residual stockout risk.",
    },
    {
        "q": "Demand is perfectly constant at 50 units/day and lead time is always exactly 3 days. Do I need safety stock?",
        "a": "No. With zero variability in both demand and lead time, the reorder point is exactly 150 units (50 × 3) and safety stock = 0. Safety stock compensates for variability; if there is none, it adds cost without benefit.",
    },
]

prompts = {
    "Version 1": "You are an inventory management advisor. Answer the question.",
    "Version 2": """You are an inventory management advisor for warehouse operations teams.

    When answering questions:
    1. State the core inventory principle or formula first (e.g., EOQ, ROP, safety stock)
    2. Explain BOTH the textbook/ideal case AND the real-world complication if they differ
    3. Explicitly correct any common misconception embedded in the question
    4. Use concrete numbers from the question to illustrate your answer

    Always address the specific edge case or nuance — managers ask these questions because they have a hidden assumption that may be wrong.""",
}


def mark(score):
    return "✅" if score else "❌"


def bar(passed, total, width=20):
    filled = round(width * passed / total) if total else 0
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    # ---------- Step 1: look at one question side by side ----------
    q1 = dataset[0]["q"]
    print(f"\nQUESTION 1:\n{q1}\n")
    print("=" * 70)
    for prompt_name in prompts:
        print(f"\n{prompt_name} ANSWER:")
        print("-" * 30)
        print(inventory_advisor(prompts[prompt_name], q1))

    # ---------- Step 2: run every question through both prompts and judge each answer ----------
    results = {name: {m: [] for m in METRICS} for name in prompts}
    rows = []
    for prompt_name, prompt in prompts.items():
        print(f"\n{'=' * 70}\nEvaluating: {prompt_name}\n{'=' * 70}")
        print(f"  {'#':<4}{'Acc':^6}{'Clar':^6}{'Comp':^6}  Judge's reason")
        for i, item in enumerate(dataset, start=1):
            response = inventory_advisor(prompt, item["q"])
            scores = evaluate_response(item["q"], item["a"], response)
            for m in METRICS:
                results[prompt_name][m].append(scores[m])
            print(f"  Q{i:<3}{mark(scores['Accuracy']):^5}{mark(scores['Clarity']):^5}"
                  f"{mark(scores['Completeness']):^5}  {scores['Reason']}")
            rows.append({"prompt_version": prompt_name, "question": item["q"], "expected": item["a"],
                         "response": response, **{m.lower(): scores[m] for m in METRICS},
                         "judge_reason": scores["Reason"]})

    # ---------- Step 3: pass rates ----------
    print("\n" + "=" * 70)
    print("PASS RATES BY PROMPT VERSION")
    print("=" * 70)
    for prompt_name in prompts:
        print(f"\n{prompt_name.upper()}:")
        total_pass = total_tests = 0
        for m in METRICS:
            passed, total = sum(results[prompt_name][m]), len(results[prompt_name][m])
            total_pass, total_tests = total_pass + passed, total_tests + total
            print(f"  {m:<13} {bar(passed, total)} {passed}/{total} ({100 * passed / total:.0f}%)")
        print(f"  {'Overall':<13} {bar(total_pass, total_tests)} {total_pass}/{total_tests} "
              f"({100 * total_pass / total_tests:.0f}%)")

    # ---------- Step 4: save every answer for human review (the annotator reads this file) ----------
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} judged answers to {RESULTS_CSV}")
