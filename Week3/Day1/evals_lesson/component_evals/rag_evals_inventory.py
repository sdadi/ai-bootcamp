"""
RAG evaluation: which retrieval + prompt setup answers inventory questions best?
By Shivank Agarwal. Models: OpenAI gpt-6-luna (answers and judge), text-embedding-3-small (search).

1. A small knowledge base: 12 inventory-policy passages.
2. A simple RAG agent: embed the question, fetch the k closest passages, answer from them.
3. A golden dataset: 10 questions, each with a reference answer and the passages it needs.
4. Three configurations: A (k=1, minimal prompt), B (k=2, structured), C (k=3, grounded).
5. Two kinds of metric:
   - Retrieval Recall@k (no model needed): did we fetch the passages the question needs?
   - LLM-as-judge (0/1): Context Relevance, Answer Groundedness, Answer Correctness.
6. A comparison table, and a CSV of every answer for human review.
"""
import csv
import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                                        # reads OPENAI_API_KEY from .env
client = OpenAI()
MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"
NUM_QUESTIONS = 5                                    # set to 10 for the full run
METRICS = ["Context Relevance", "Answer Groundedness", "Answer Correctness"]
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_eval_results.csv")


def evaluate_rag_response(question, retrieved_context, generated_answer, reference_answer):
    """LLM-as-judge: 3 binary scores (0/1) plus a one-line reason."""
    eval_prompt = f"""You are evaluating a RAG system for inventory management. Be STRICT with scoring.

    Question: {question}
    Retrieved Context: {retrieved_context}
    Generated Answer: {generated_answer}
    Reference Answer: {reference_answer}

    Evaluate these 3 metrics:

    1. Context Relevance (0 or 1):
    Does the retrieved context contain ALL information needed to fully answer the question?
    Score 0 if the context is missing key facts mentioned in the reference answer.

    2. Answer Groundedness (0 or 1):
    Is EVERY claim in the generated answer explicitly supported by the retrieved context?
    Score 0 if the answer includes ANY information not found in the context.
    Score 0 if the answer adds details or numbers beyond what the context provides.

    3. Answer Correctness (0 or 1):
    Does the generated answer match the reference answer's key facts?
    Score 0 if any key information is wrong or missing.

    Respond ONLY with JSON in this exact shape:
    {{"Context Relevance": 0 or 1, "Answer Groundedness": 0 or 1, "Answer Correctness": 0 or 1, "Reason": "one short sentence"}}"""

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


documents = [
    # Doc 0: Reorder point basics
    "A reorder point (ROP) is the inventory level that triggers a new purchase order. "
    "It is calculated as: ROP = average daily demand × lead time in days.",

    # Doc 1: Safety stock (needs Doc 0 for full picture)
    "Safety stock is extra inventory held to guard against demand spikes and lead-time variability. "
    "The full reorder point formula is: ROP = (average demand × lead time) + safety stock. "
    "Without safety stock the formula only works when demand and lead time are perfectly constant.",

    # Doc 2: EOQ definition
    "Economic Order Quantity (EOQ) is the order size that minimises total inventory cost. "
    "Formula: EOQ = sqrt(2 × D × S / H), where D = annual demand, S = ordering cost, H = holding cost per unit per year.",

    # Doc 3: EOQ sensitivity (needs Doc 2)
    "In the EOQ formula, holding cost H is in the denominator under the square root. "
    "When holding cost doubles, EOQ decreases — it becomes cheaper to order smaller quantities more frequently. "
    "Conversely, when ordering cost S rises, EOQ increases.",

    # Doc 4: ABC classification basics
    "ABC analysis classifies inventory by value: A items are high-value (top 70-80% of spend), "
    "B items are medium-value, and C items are low-value (bottom 5% of spend). "
    "Cycle count frequency and review effort are typically proportional to class.",

    # Doc 5: ABC criticality caveat (needs Doc 4)
    "ABC classification is based on monetary value, not operational importance. "
    "A low-value (Class C) item can still be operationally critical — for example, a cheap O-ring "
    "that stops an entire production line. Critical C items require tighter controls despite low cost.",

    # Doc 6: Lead time and supplier trade-offs
    "Supplier lead time directly affects required safety stock and reorder point. "
    "A supplier with a longer lead time forces higher safety stock, increasing holding costs. "
    "Total procurement cost = unit price + ordering cost + holding cost; a higher-priced supplier "
    "with shorter lead time may have lower total cost than a cheaper supplier with long lead time.",

    # Doc 7: Service level and stockout risk (needs Doc 1)
    "A 100% service level (zero stockout) would require infinite safety stock, which is not economically viable. "
    "Companies choose a target service level (e.g., 95% or 99%) and calculate the safety stock needed to achieve it "
    "using the formula: safety stock = Z × σ_demand × sqrt(lead time), where Z is the service-level z-score.",

    # Doc 8: Perishable inventory
    "Perishable items have a finite shelf life. The reorder cycle must be short enough that items are consumed "
    "before expiry. FEFO (First Expired First Out) rotation ensures oldest stock is picked first. "
    "If the reorder cycle approaches the shelf life, there is a risk of expiry before consumption.",

    # Doc 9: In-transit inventory
    "In-transit (pipeline) inventory refers to stock already ordered but not yet received. "
    "When calculating whether to reorder, on-hand inventory plus in-transit inventory should both be compared "
    "against the reorder point. Reordering when in-transit stock will bring inventory above ROP causes overstock.",

    # Doc 10: Demand variability and averaging
    "A single week of unusually high demand should not immediately trigger a permanent change to the reorder point. "
    "Demand should be averaged over a rolling window (typically 4–8 weeks) to smooth out anomalies. "
    "Reacting to a single spike risks permanent overstock if the spike was seasonal or one-off.",

    # Doc 11: Zero-variability special case (needs Doc 1 and Doc 7)
    "When both demand and lead time are perfectly constant with zero variability, safety stock is zero. "
    "The reorder point equals exactly: average demand × lead time. "
    "Safety stock exists to absorb variability; if there is none, safety stock adds cost without benefit.",
]

# ---------- Embeddings: every passage is embedded ONCE, in a single API call ----------
def embed(texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])


doc_embeddings = embed(documents)                    # shape: (12, 1536)


def retrieve(query, k=2):
    """Return the numbers of the k passages closest in meaning to the query."""
    q = embed([query])[0]
    sims = doc_embeddings @ q / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q))
    return [int(i) for i in np.argsort(sims)[::-1][:k]]      # highest cosine similarity first


PROMPT_MINIMAL = """Context: {context}

Question: {question}
Answer:"""

PROMPT_STRUCTURED = """Use the context below to answer the inventory question.

Context:
{context}

Question: {question}
Answer based on the context:"""

PROMPT_GROUNDED = """You are an inventory management advisor. Answer using ONLY the provided context.
Combine information from all context passages to give a complete answer.
If the context is incomplete, say what's missing.

Context:
{context}

Question: {question}
Answer (use all relevant context):"""


def rag_agent(question, k=2, prompt_template=PROMPT_MINIMAL):
    """RAG agent: retrieve k passages, then answer from them."""
    doc_ids = retrieve(question, k=k)
    context = "\n---\n".join(documents[i] for i in doc_ids)
    prompt = prompt_template.format(context=context, question=question)
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        messages=[{"role": "user", "content": prompt}],
    )
    return {"doc_ids": doc_ids, "context": context, "answer": response.choices[0].message.content}


eval_dataset = [
    {
        "question": "What is the reorder point formula and when should safety stock be included?",
        "reference": "ROP = average demand × lead time. Safety stock must be added when demand or lead time vary: ROP = (avg demand × lead time) + safety stock.",
        "relevant_docs": [0, 1],
    },
    {
        "question": "If our holding cost doubles, should we increase or decrease our order quantity?",
        "reference": "Decrease. In EOQ = sqrt(2DS/H), H is in the denominator. Doubling H reduces EOQ — smaller, more frequent orders become cheaper.",
        "relevant_docs": [2, 3],
    },
    {
        "question": "A Class C item has very low cost. Can we safely ignore it in cycle counts?",
        "reference": "Not always. ABC is value-based, not criticality-based. A cheap Class C item can be operationally critical and stop a production line. Critical C items need tighter controls.",
        "relevant_docs": [4, 5],
    },
    {
        "question": "Supplier A is cheaper per unit but has a 30-day lead time. Supplier B costs more but delivers in 5 days. How should we evaluate them?",
        "reference": "Compare total cost = unit price + ordering cost + holding cost. Longer lead time forces higher safety stock and higher holding costs. Supplier B may have lower total cost despite higher unit price.",
        "relevant_docs": [6, 1],
    },
    {
        "question": "Can we set safety stock high enough to guarantee we never stock out?",
        "reference": "No. 100% service level requires infinite safety stock. Choose a target service level (e.g., 95% or 99%) and calculate safety stock using: safety stock = Z × σ_demand × sqrt(lead time).",
        "relevant_docs": [7, 1],
    },
    {
        "question": "We have a perishable product with a 10-day shelf life and reorder every 8 days. Is there a risk?",
        "reference": "Yes. If stock arrives and sits near the end of the cycle, remaining shelf life at time of use could be very short. Use FEFO rotation and ensure the reorder cycle is well within shelf life.",
        "relevant_docs": [8],
    },
    {
        "question": "We have 220 units on hand and a PO of 100 units arriving tomorrow. ROP is 200. Should we reorder?",
        "reference": "No. On-hand (220) already exceeds ROP (200), and in-transit stock (100) will push inventory even higher. Reordering now causes overstock. Always include in-transit inventory in the reorder check.",
        "relevant_docs": [9, 0],
    },
    {
        "question": "Demand spiked 50% last week. Should I immediately raise the reorder point by 50%?",
        "reference": "No. A single week's spike may be seasonal or anomalous. Average demand over a rolling 4–8 week window before adjusting the reorder point. Reacting to one spike risks permanent overstock.",
        "relevant_docs": [10, 0],
    },
    {
        "question": "Our demand is exactly 40 units/day and lead time is always exactly 5 days. Do we need safety stock?",
        "reference": "No. With zero variability in both demand and lead time, ROP = 40 × 5 = 200 units and safety stock = 0. Safety stock compensates for variability; with none, it only adds holding cost.",
        "relevant_docs": [11, 1],
    },
    {
        "question": "What is EOQ and what happens to it when ordering cost increases?",
        "reference": "EOQ = sqrt(2DS/H). When ordering cost S increases, EOQ increases — it becomes cheaper to order larger quantities less frequently to spread the fixed ordering cost.",
        "relevant_docs": [2, 3],
    },
]

rag_configs = {
    "Config A: k=1, minimal":    {"k": 1, "prompt": PROMPT_MINIMAL},
    "Config B: k=2, structured": {"k": 2, "prompt": PROMPT_STRUCTURED},
    "Config C: k=3, grounded":   {"k": 3, "prompt": PROMPT_GROUNDED},
}


def recall_at_k(retrieved, relevant):
    """Share of the needed passages that were actually retrieved (1.0 = all of them)."""
    return len(set(retrieved) & set(relevant)) / len(relevant)


def mark(score):
    return "✅" if score else "❌"


if __name__ == "__main__":
    print(f"Corpus: {len(documents)} passages, embedded with {EMBED_MODEL}")
    print(f"Evaluation dataset: {len(eval_dataset)} questions (running the first {NUM_QUESTIONS})")
    print("RAG configurations:")
    for name, cfg in rag_configs.items():
        print(f"  {name} (retrieves {cfg['k']} passage(s))")

    test_data = eval_dataset[:NUM_QUESTIONS]
    results = {name: {m: [] for m in METRICS + ["Recall@k"]} for name in rag_configs}
    rows = []

    for config_name, config in rag_configs.items():
        print(f"\n{'=' * 78}\nEvaluating: {config_name}\n{'=' * 78}")
        print(f"  {'#':<4}{'needed':<9}{'retrieved':<12}{'Recall':>7}  {'Rel':^5}{'Grd':^5}{'Cor':^5}  Judge's reason")
        for i, item in enumerate(test_data, start=1):
            out = rag_agent(item["question"], k=config["k"], prompt_template=config["prompt"])
            recall = recall_at_k(out["doc_ids"], item["relevant_docs"])
            scores = evaluate_rag_response(item["question"], out["context"], out["answer"], item["reference"])
            for m in METRICS:
                results[config_name][m].append(scores[m])
            results[config_name]["Recall@k"].append(recall)
            print(f"  Q{i:<3}{str(item['relevant_docs']):<9}{str(out['doc_ids']):<12}{recall:>7.0%}  "
                  f"{mark(scores['Context Relevance']):^4}{mark(scores['Answer Groundedness']):^4}"
                  f"{mark(scores['Answer Correctness']):^4}  {scores['Reason']}")
            rows.append({"config": config_name, "question": item["question"], "reference": item["reference"],
                         "relevant_docs": item["relevant_docs"], "retrieved_docs": out["doc_ids"],
                         "recall_at_k": round(recall, 2), "response": out["answer"],
                         **{m.lower().replace(" ", "_"): scores[m] for m in METRICS},
                         "judge_reason": scores["Reason"]})

    print("\n" + "=" * 78)
    print(f"RAG EVALUATION RESULTS ({NUM_QUESTIONS} questions per configuration)")
    print("=" * 78)
    print(f"{'Configuration':<28}{'Recall@k':>10}{'Relevant':>10}{'Grounded':>10}{'Correct':>10}{'Overall':>10}")
    print("-" * 78)
    best_config, best_score = None, -1
    for config_name in rag_configs:
        r = results[config_name]
        avg = {m: sum(v) / len(v) for m, v in r.items()}
        overall = (avg["Context Relevance"] + avg["Answer Groundedness"] + avg["Answer Correctness"]) / 3
        print(f"{config_name:<28}{avg['Recall@k']:>10.0%}{avg['Context Relevance']:>10.0%}"
              f"{avg['Answer Groundedness']:>10.0%}{avg['Answer Correctness']:>10.0%}{overall:>10.0%}")
        if overall > best_score:
            best_score, best_config = overall, config_name
    print("-" * 78)
    print(f"Best: {best_config} ({best_score:.0%})")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} judged answers to {RESULTS_CSV}")
