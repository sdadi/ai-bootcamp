"""
trip_03_chromadb_retrieval.py
─────────────────────────────
Lesson 03 · ChromaDB: a proper "library" for our tips, with labels (metadata).

The story
    The travel notebook grows. Instead of a Python list, we put every tip in a
    vector database (ChromaDB). Each tip also gets a label, like a shelf in a
    library: "money", "trip_type" or "travel_tip".

What you learn
    - ChromaDB stores text + its embedding + metadata together
    - Query by meaning AND filter by metadata (e.g. only topic = "money")
    - retrieve() is the building block every RAG system uses
    - Still no LLM, only retrieval

How to run
    uv run trip_03_chromadb_retrieval.py
    (no API key needed)

What you will see
    Three searches: one open search, one limited to money tips, one limited to trip types.
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Same free model as lesson 02, wrapped by LangChain so it plugs into other tools.
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# In-memory Chroma client. "hnsw:space": "cosine" = compare vectors by cosine similarity.
client = chromadb.Client()
collection = client.create_collection("trip_tips", metadata={"hnsw:space": "cosine"})

# 8 tips, each with a shelf label in "meta"
docs = [
    {"id": "trip-cost",  "text": "Trip cost per person = travel + stay share + daily spend x nights.",                              "meta": {"topic": "money"}},
    {"id": "rooms",      "text": "One room fits 2 people, so rooms needed = people / 2, rounded up. Stay share = rooms x room price x nights / people.", "meta": {"topic": "money"}},
    {"id": "beach",      "text": "Beach trips (Goa, Gokarna, Pondicherry): best November-February. In the monsoon many beach shacks and water sports close.", "meta": {"topic": "trip_type"}},
    {"id": "hills",      "text": "Hill trips (Coorg, Ooty, Manali): best March-June and October. Avoid July-August: heavy rain and landslides.", "meta": {"topic": "trip_type"}},
    {"id": "heritage",   "text": "Heritage trips (Hampi, Jaipur): old temples, forts and palaces. Best October-March; April-May afternoons cross 40°C.", "meta": {"topic": "trip_type"}},
    {"id": "adventure",  "text": "Adventure trips (Rishikesh, Dandeli): river rafting runs October-June and closes in the monsoon.", "meta": {"topic": "trip_type"}},
    {"id": "book-early", "text": "Book trains and flights 30-60 days early; prices often double in the last week.",                  "meta": {"topic": "travel_tip"}},
    {"id": "weekdays",   "text": "Weekday stays are often 20-30% cheaper than weekends.",                                            "meta": {"topic": "travel_tip"}},
]

print("Embedding and storing documents...")
vectors = embed.embed_documents([d["text"] for d in docs])
collection.add(
    ids=[d["id"] for d in docs],
    embeddings=vectors,
    documents=[d["text"] for d in docs],
    metadatas=[d["meta"] for d in docs],
)
print(f"Collection has {collection.count()} documents\n")


def retrieve(query: str, n: int = 3, where: dict | None = None) -> list[str]:
    """Find the n tips closest in meaning to `query`.

    where = optional metadata filter, e.g. {"topic": "money"}
            means "only look on the money shelf".
    """
    qv = embed.embed_query(query)
    kwargs = {"query_embeddings": [qv], "n_results": n}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return results["documents"][0]


print("=== Query: 'Manali trip in the monsoon' ===")
for chunk in retrieve("Manali trip in the monsoon"):
    print(" •", chunk)

print()

print("=== Query: 'how do we split the cost' (money tips only) ===")
for chunk in retrieve("how do we split the cost", where={"topic": "money"}):
    print(" •", chunk)

print()

print("=== Query: 'old temples and forts' (trip types only) ===")
for chunk in retrieve("old temples and forts", where={"topic": "trip_type"}):
    print(" •", chunk)
