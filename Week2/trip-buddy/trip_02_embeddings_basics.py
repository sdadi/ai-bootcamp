"""
trip_02_embeddings_basics.py
────────────────────────────
Lesson 02 · Embeddings: turn sentences into numbers and find the closest match.

The story
    Your most-travelled friend keeps a notebook of travel tips. When someone asks
    "Is it safe to go to the mountains in the rainy season?", we want the tip that
    MEANS the same thing ("Hills: avoid July-August, landslides"), even though
    the words are different.

What you learn
    - A sentence-transformer model turns text into 384 numbers (an embedding)
    - Sentences with similar meaning get vectors pointing in a similar direction
    - Cosine similarity = dot product of two unit-length vectors (1.0 = same meaning)
    - No database and no LLM yet, only numpy and a small free model

How to run
    uv run trip_02_embeddings_basics.py
    (no API key needed; the first run downloads the ~90 MB model once)

What you will see
    For each question, the top 3 tips with a similarity score.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# The travel notebook
PASSAGES = [
    "Trip cost per person = travel + stay share + daily spend x nights.",
    "One room fits 2 people, so rooms needed = people / 2, rounded up.",
    "Hills (Coorg, Ooty, Manali): avoid July-August, heavy rain and landslides. Carry a jacket.",
    "Beach (Goa, Gokarna, Pondicherry): best November-February; many shacks close in the monsoon.",
    "Heritage (Hampi, Jaipur): explore early in the morning, afternoons get very hot in April-May.",
    "Book trains 30-60 days early; last-week prices can double.",
]

# A small, free, local model. Each sentence becomes a vector of 384 numbers.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding passages...")
# normalize_embeddings=True makes every vector length 1,
# so a plain dot product IS the cosine similarity.
passage_vectors = model.encode(PASSAGES, normalize_embeddings=True)
print(f"Passage matrix shape: {passage_vectors.shape}\n")   # (6 passages, 384 numbers)


def top_k(query: str, k: int = 3) -> list[tuple[float, str]]:
    """Return the k passages most similar in meaning to the query."""
    qv = model.encode(query, normalize_embeddings=True)
    scores = passage_vectors @ qv              # one score per passage, shape: (6,)
    ranked = np.argsort(scores)[::-1][:k]      # indexes of the highest scores first
    return [(float(scores[i]), PASSAGES[i]) for i in ranked]


queries = [
    "Where should I go to see the sea in winter?",
    "Is it safe to go to the mountains in the rainy season?",
    "How do we split hotel rooms between friends?",
]

for query in queries:
    print(f"Query: {query!r}")
    for score, passage in top_k(query):
        print(f"  {score:.3f}  {passage}")
    print()
