"""
LAB 01 - A complete RAG system in ~60 lines  (Module 2)
=======================================================
What you will learn
  * The two pipelines of every RAG system:
      INDEXING  (offline): load -> chunk -> embed -> store
      QUERYING  (online):  embed the question -> find similar chunks -> build a prompt -> LLM
  * That a "naive" RAG already works on clean documents... and where it starts to break.

How to run
    python labs/lab01_rag_in_60_lines.py
    python labs/lab01_rag_in_60_lines.py "Can I pay cash on delivery for a 7000 rupee order?"

What to look for
  * The top-3 chunks and their cosine similarity scores.
  * The exact prompt sent to the LLM (this is all the LLM ever sees).
  * Try "What does error E102 mean?" - it fails here because the naive index only has two
    files and no keyword search. Later labs fix both problems.
"""
import sys

import numpy as np

import _common

question = sys.argv[1] if len(sys.argv) > 1 else "How many days do I have to return a mobile phone?"

# ---------------- INDEXING (offline) -------------------------------------------
files = ["data/raw/refund_policy_2025.md", "data/raw/delivery_and_shipping.md"]
chunks = []
for f in files:
    text = open(f, encoding="utf-8").read()
    # naive chunking: one chunk per paragraph (blank-line separated)
    chunks += [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

embedder = _common.embedder_or_exit()
vectors = embedder.embed_documents(chunks)           # shape: (number_of_chunks, dimensions)
print(f"Indexed {len(chunks)} chunks with {embedder.name} -> matrix {vectors.shape}")

# ---------------- QUERYING (online) --------------------------------------------
q = embedder.embed_query(question)                    # same model as the documents!
scores = vectors @ q                                  # cosine similarity (vectors have length 1)
top = np.argsort(-scores)[:3]

print(f"\nQuestion: {question}\n\nTop 3 chunks:")
for rank, i in enumerate(top, start=1):
    print(f"  {rank}. score={scores[i]:.3f}  {' '.join(chunks[i].split())[:110]}...")

context = "\n\n".join(f"[S{r}] {chunks[i]}" for r, i in enumerate(top, start=1))
prompt = (f"Answer the question using ONLY the sources below. Cite sources like [S1]. "
          f"If the answer is not in the sources, say you don't know.\n\n{context}\n\nQuestion: {question}")
print("\n----- PROMPT SENT TO THE LLM -----\n" + prompt + "\n----------------------------------")

llm = _common.llm_status()
if llm.available:
    result = llm.generate("You are a helpful assistant.", prompt)
    print(f"\nANSWER ({result.latency_ms:.0f} ms, {result.input_tokens} in / {result.output_tokens} out tokens):\n{result.text}")
else:
    print("\nNo LLM key, so here is the best-matching chunk as the 'answer':\n ", chunks[top[0]][:300])
