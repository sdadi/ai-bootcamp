"""
Step 03 — ChromaDB: a real vector database with metadata filters
Created by Shivank Agarwal

Goal: store passages + their vectors + tags (metadata) in ChromaDB, then search by meaning
and optionally filter by tag, e.g. "only search the formulas".

Run it:
    python 03_chromadb_retrieval.py
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Same local model as Step 02, wrapped by LangChain
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

client = chromadb.Client()  # in-memory ChromaDB (nothing saved to disk)
collection = client.create_collection("inventory_kb", metadata={"hnsw:space": "cosine"})

docs = [
    {"id": "ss-formula",  "text": "Safety Stock = Z * demand_std_dev * sqrt(lead_time). Z=1.65 for 95%, Z=2.33 for 99%.",   "meta": {"topic": "formula"}},
    {"id": "rop-formula", "text": "Reorder Point (ROP) = (avg_daily_demand * lead_time) + safety_stock.",                   "meta": {"topic": "formula"}},
    {"id": "electronics", "text": "Electronics: lead times 14-30 days, service level 95-99%, high obsolescence risk.",      "meta": {"topic": "category"}},
    {"id": "pharma",      "text": "Pharma: lead times 7-21 days, service level 99%+, strict FIFO and expiry management.",   "meta": {"topic": "category"}},
    {"id": "fmcg",        "text": "FMCG: lead times 3-7 days, high steady demand, service level 95-98%.",                   "meta": {"topic": "category"}},
    {"id": "raw-mat",     "text": "Raw materials: lead times 10-45 days, service level 90-95%.",                            "meta": {"topic": "category"}},
    {"id": "abc",         "text": "ABC Analysis: A-items need tight control and high service levels.",                      "meta": {"topic": "best_practice"}},
    {"id": "variability", "text": "High demand variability requires more safety stock to maintain service level.",          "meta": {"topic": "best_practice"}},
]

print("Embedding and storing documents...")
# ----- TODO 1: Embed every text and add ids, vectors, texts and metadata to the collection -----
# Hint: vectors = embed.embed_documents([d["text"] for d in docs])
# Hint: collection.add(ids=[...], embeddings=vectors, documents=[...], metadatas=[...])
pass  # replace with your code
print(f"Collection has {collection.count()} documents\n")


def retrieve(query: str, n: int = 3, where: dict | None = None) -> list[str]:
    """Return the n most similar texts. `where` is an optional metadata filter, e.g. {"topic": "formula"}."""
    # ----- TODO 2: Embed the query and search the collection (apply the filter only if one is given) -----
    # Hint: qv = embed.embed_query(query)
    # Hint: collection.query(query_embeddings=[qv], n_results=n, where=where)
    # Hint: Only add the where filter when one is given (it is None by default)
    # Hint: The texts are in results["documents"][0]
    return []  # replace with your code


print("=== Query: 'pharma reorder safety stock' ===")
for chunk in retrieve("pharma reorder safety stock"):
    print(" •", chunk)

print("\n=== Query: 'how to calculate stock' (formulas only) ===")
for chunk in retrieve("how to calculate stock", where={"topic": "formula"}):
    print(" •", chunk)

print("\n=== Query: 'electronics inventory management' (categories only) ===")
for chunk in retrieve("electronics inventory management", where={"topic": "category"}):
    print(" •", chunk)
