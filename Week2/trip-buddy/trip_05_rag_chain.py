"""
trip_05_rag_chain.py
────────────────────
Lesson 05 · RAG: give the AI the travel notebook before it answers.

The story
    Three friends want to go to Manali in July. We ask the AI planner twice:
      - WITHOUT RAG: it answers from general knowledge only.
      - WITH RAG: we first look up the most relevant tips (retrieve), paste them
        into the prompt (augment), then let it answer (generate).
    Does it warn us about July rains and landslides?

What you learn
    - Retrieve → Augment → Generate, the core RAG pattern
    - {rag_context} is just text pasted into the prompt
    - Why grounded answers are more specific and more useful

How to run
    uv run trip_05_rag_chain.py        (needs OPENAI_API_KEY in .env)

What you will see
    The retrieved tips, then the answer WITHOUT RAG, then the answer WITH RAG.
"""

import os

import chromadb
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 1. Build the vector store (same idea as lesson 03) ────────────────────────
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

client = chromadb.Client()
col = client.create_collection("trip_tips", metadata={"hnsw:space": "cosine"})

KB = [
    ("cost",   "Trip cost per person = travel + stay share + daily spend x nights."),
    ("rooms",  "One room fits 2 people, so rooms needed = people / 2, rounded up."),
    ("hills",  "Hill trips (Coorg, Ooty, Manali): best March-June and October. Avoid July-August: heavy rain and landslides block hill roads."),
    ("beach",  "Beach trips: best November-February; many shacks close in the monsoon."),
    ("buffer", "Keep a 10% buffer on top of the total for surprises like a cab surge or a missed bus."),
    ("pack",   "Mountain weather changes fast: carry a rain jacket and warm layers even in summer."),
]
vecs = embed.embed_documents([t for _, t in KB])
col.add(ids=[i for i, _ in KB], embeddings=vecs, documents=[t for _, t in KB])


def retrieve_context(query: str, n: int = 3) -> str:
    """RETRIEVE step: find the n most relevant tips and return them as a bullet list."""
    qv = embed.embed_query(query)
    res = col.query(query_embeddings=[qv], n_results=n)
    return "\n".join(f"- {c}" for c in res["documents"][0])


# ── 2. Chain WITHOUT RAG ──────────────────────────────────────────────────────
bare_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a trip planner for a group of friends. Give advice on this trip plan."),
    ("human",
     "Place: {place} ({trip_type}) | Month: {month}\n"
     "Friends: {people} | Nights: {nights} | Budget per person: ₹{budget_per_person}"),
])

# ── 3. Chain WITH RAG (only difference: {rag_context} in the system message) ──
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a trip planner for a group of friends. Give advice on this trip plan.\n\n"
     "Relevant travel tips:\n{rag_context}"),     # ← AUGMENT step
    ("human",
     "Place: {place} ({trip_type}) | Month: {month}\n"
     "Friends: {people} | Nights: {nights} | Budget per person: ₹{budget_per_person}"),
])

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
llm = ChatOpenAI(model=MODEL, reasoning_effort="none")   # see lesson 04 for why
parser = StrOutputParser()

bare_chain = bare_prompt | llm | parser
rag_chain  = rag_prompt  | llm | parser                   # ← GENERATE step

# ── 4. Run the comparison ─────────────────────────────────────────────────────
trip = {
    "place":             "Manali",
    "trip_type":         "Hills",
    "month":             "July",
    "people":            3,
    "nights":            4,
    "budget_per_person": 20000,
}

# Build a search query from the trip details, then retrieve matching tips
query = f"{trip['trip_type']} trip to {trip['place']} in {trip['month']}"
context = retrieve_context(query)

print("=== Retrieved Context ===")
print(context)

print("\n=== Response WITHOUT RAG ===")
print(bare_chain.invoke(trip))

print("\n=== Response WITH RAG ===")
print(rag_chain.invoke({**trip, "rag_context": context}))
