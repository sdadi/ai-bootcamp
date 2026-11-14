"""
Step 02 — Embeddings basics: find the most relevant passage for a question
Created by Shivank Agarwal

Goal: turn short inventory notes into vectors and find the best ones for a question.
No database and no LLM yet: just an embedding model + numpy.

Run it:
    python 02_embeddings_basics.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Our tiny "knowledge base": six facts about inventory planning
PASSAGES = [
    "Safety Stock = Z-score * demand_std_dev * sqrt(lead_time_days)",
    "Reorder Point = (avg_daily_demand * lead_time_days) + safety_stock",
    "Electronics: lead times 14-30 days, high obsolescence risk.",
    "Pharma: strict FIFO, 99%+ service level, expiry management.",
    "FMCG: short lead times 3-7 days, high steady demand.",
    "ABC Analysis: A-items need tight control, C-items tolerate bulk orders.",
]

# Free local model: every text -> 384 numbers
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding passages...")
# normalize_embeddings=True makes every vector length 1,
# so cosine similarity becomes a simple dot product
passage_vectors = model.encode(PASSAGES, normalize_embeddings=True)
print(f"Passage matrix shape: {passage_vectors.shape}  (6 passages x 384 numbers)\n")


def top_k(query: str, k: int = 3) -> list[tuple[float, str]]:
    """Return the k passages most similar to the query as (score, passage), best first."""
    # ----- TODO 1: Embed the query, score it against every passage, and return the top k -----
    # Hint: qv = model.encode(query, normalize_embeddings=True)
    # Hint: scores = passage_vectors @ qv    (the @ operator = dot product for every row at once)
    # Hint: np.argsort(scores)[::-1][:k] = indexes of the k highest scores
    return []  # replace with your code


queries = [
    "How do I calculate safety stock?",
    "What should I know about electronics inventory?",
    "Tell me about demand patterns for consumer goods.",
]

for query in queries:
    print(f"Query: {query!r}")
    for score, passage in top_k(query):
        print(f"  {score:.3f}  {passage}")
    print()
