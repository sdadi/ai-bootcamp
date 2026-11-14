"""
LAB 04 - Embeddings: meaning as numbers  (Module 7)
===================================================
What you will learn
  * Embeddings put sentences with the same MEANING close together, even with different words.
  * Cosine, dot product and Euclidean distance give the SAME ranking on normalised vectors.
  * Storage maths: dimensions x bytes x number of chunks.
  * Quantization (int8, binary) and shortening vectors (Matryoshka) - and what they cost in recall.
  * Why an English-only model fails on Hindi questions.

How to run
    python labs/lab04_embeddings.py

What to look for
  * In the similarity matrix, "refund" and "money back" are close for a REAL model but not for
    the toy keyword hasher.
  * The recall table: int8 keeps almost all quality at 1/4 of the memory; binary loses more,
    and "binary + rescoring" wins most of it back.
"""
import numpy as np

import _common
from rag_core.chunking import chunk_documents
from rag_core.embeddings import HashingEmbedder, normalize
from rag_core.evaluation import is_relevant, load_golden
from rag_core.loaders import load_corpus
from rag_core.utils import banner, table

embedder = _common.embedder_or_exit()
print("Model:", embedder.name, "| dimensions:", embedder.dim)

banner("1. Same meaning, different words")
sents = [
    "When will I get my money back?",
    "Refunds to a credit card take 5 to 7 business days.",
    "How long is delivery to Jaipur?",
    "Standard shipping to Tier-2 cities takes 3 to 4 days.",
    "What is the salary band for L5?",
]
for emb in [embedder, HashingEmbedder()]:
    v = emb.embed_documents(sents)
    sim = v @ v.T
    print(f"\n{emb.name}: cosine similarity of every pair")
    print("      " + "  ".join(f"S{i+1:<4}" for i in range(len(sents))))
    for i, row in enumerate(sim):
        print(f"S{i+1}  " + "  ".join(f"{x:5.2f}" for x in row) + f"   {sents[i][:45]}")
print("\nLook at S1-S2 (refund question vs refund policy) and S3-S4 (delivery question vs shipping rule).")
print("A real embedding model scores these pairs high although they share few words. The hasher cannot.")

banner("2. Cosine vs dot product vs Euclidean")
rng = np.random.default_rng(0)
docs = normalize(rng.normal(size=(1000, 64)))
q = normalize(rng.normal(size=64))
by_cos = np.argsort(-(docs @ q) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q)))[:10]
by_dot = np.argsort(-(docs @ q))[:10]
by_l2 = np.argsort(np.linalg.norm(docs - q, axis=1))[:10]
print("Normalised vectors -> identical top-10?  cos==dot:", (by_cos == by_dot).all(), " cos==L2:", (by_cos == by_l2).all())
raw = docs * rng.uniform(0.5, 3, size=(1000, 1))  # same directions, random lengths
print("Un-normalised vectors -> cos==dot:", (np.argsort(-(raw @ q))[:10] == by_cos).all(),
      "(dot product now favours LONG vectors, not similar ones)")
print("Rule: ||a-b||^2 = 2 - 2*cos(a,b) when ||a|| = ||b|| = 1, so all three agree after normalising.")

banner("3. How much memory will my index need?")
rows = []
for dim in [384, 768, 1024, 3072]:
    for fmt, bytes_per in [("float32", 4), ("float16", 2), ("int8", 1), ("binary", 1 / 8)]:
        rows.append({"dims": dim, "format": fmt, "per vector": f"{dim * bytes_per:,.0f} B",
                     "1 million chunks": f"{dim * bytes_per * 1e6 / 1e9:,.2f} GB"})
table(rows)
print("(Graph indexes like HNSW add memory on top: roughly M*2*4 bytes of links per vector.)")

banner("4. Quantization: smaller vectors, how much search quality do we lose?")
# Our toy corpus has only ~30 chunks, too few to see differences. So we use 20,000 synthetic
# clustered vectors (shaped like real embeddings) and measure recall@10 against exact float32 search.
rng = np.random.default_rng(7)
dim, n_docs, n_q, k = embedder.dim, 20_000, 200, 10
centres = rng.normal(size=(100, dim))
mat = normalize(centres[rng.integers(0, 100, n_docs)] + 0.7 * rng.normal(size=(n_docs, dim)))
qmat = normalize(centres[rng.integers(0, 100, n_q)] + 0.7 * rng.normal(size=(n_q, dim)))
truth = np.argsort(-(qmat @ mat.T), axis=1)[:, :k]


def recall_at_10(scores):
    found = np.argsort(-scores, axis=1)[:, :k]
    return round(float(np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)])), 3)


scale = np.abs(mat).max()
int8 = np.round(mat / scale * 127).astype(np.int8)            # 1 byte per dimension
bits_d, bits_q = np.packbits(mat > 0, axis=1), np.packbits(qmat > 0, axis=1)  # 1 bit per dimension
hamming = np.array([np.unpackbits(bits_d ^ bq, axis=1).sum(1) for bq in bits_q])
rescored = np.full((n_q, n_docs), -1e9)
for i in range(n_q):  # stage 1: binary search keeps 100 candidates; stage 2: exact score on those only
    cand = np.argsort(hamming[i])[:100]
    rescored[i, cand] = mat[cand] @ qmat[i]
table([
    {"variant": "float32 (baseline)", "memory per vector": f"{dim * 4} B", "recall@10": 1.0},
    {"variant": "int8 scalar quantization", "memory per vector": f"{dim} B",
     "recall@10": recall_at_10(qmat @ int8.T.astype(np.float32))},
    {"variant": "binary (Hamming distance)", "memory per vector": f"{dim // 8} B", "recall@10": recall_at_10(-hamming)},
    {"variant": "binary, then rescore top 100", "memory per vector": f"{dim // 8} B (+ floats on disk)",
     "recall@10": recall_at_10(rescored)},
])
print("""int8 keeps nearly everything at 1/4 of the memory. Binary alone loses a lot, but
'binary first, then rescore a short list with full vectors' recovers most of it.
pgvector supports halfvec (float16) and binary_quantize(); Qdrant, Milvus, Weaviate and FAISS
have scalar, binary and product quantization built in.""")

step_text = "Shortening vectors (Matryoshka / MRL) on OUR model:"
corpus_chunks = chunk_documents(load_corpus(), "structure_aware")
cm = embedder.embed_documents([c.text for c in corpus_chunks])
full_top = np.argsort(-(cm @ cm.T), axis=1)[:, 1:6]
half = cm.shape[1] // 2
half_top = np.argsort(-(normalize(cm[:, :half]) @ normalize(cm[:, :half]).T), axis=1)[:, 1:6]
agree = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(full_top, half_top)])
print(f"\n{step_text} keeping the first {half} of {cm.shape[1]} dims, the top-5 neighbours agree "
      f"{agree:.0%} with the full vectors.")
print("""MRL models are TRAINED so the first dimensions carry the most meaning, so truncating them
loses little (OpenAI text-embedding-3 'dimensions' parameter, nomic-embed v1.5, Gemini embedding,
Voyage 4, Cohere Embed, Qwen3-Embedding, EmbeddingGemma). Models without MRL lose more.""")

banner("5. A Hindi question against English documents")
hindi_q = "मोबाइल फ़ोन कितने दिनों में वापस कर सकते हैं?"  # "In how many days can I return a phone?"
en_doc = "Electronics (phones, tablets, laptops): 7 days from delivery."
hi_doc = "इलेक्ट्रॉनिक्स (फ़ोन, टैबलेट, लैपटॉप) डिलीवरी के 7 दिनों के अंदर वापस किए जा सकते हैं।"
v = embedder.embed_documents([en_doc, hi_doc])
qv = embedder.embed_query(hindi_q)
print(f"similarity(Hindi question, English answer) = {float(v[0] @ qv):.3f}")
print(f"similarity(Hindi question, Hindi answer)   = {float(v[1] @ qv):.3f}")
print("""An English-only model (like bge-small-en) cannot connect the two languages well.
For Indian languages use a multilingual model, e.g. in .env:
    EMBED_MODEL=intfloat/multilingual-e5-large      (fastembed, needs 'query: '/'passage: ' prefixes - handled for you)
or BGE-M3, Qwen3-Embedding, EmbeddingGemma, Cohere/Voyage/Gemini multilingual APIs. Then re-run this lab.""")
print("\nNext: python labs/lab05_vector_indexes.py")
