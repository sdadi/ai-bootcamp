"""
Step 05 — RAG chain: Retrieve -> Augment -> Generate
Created by Shivank Agarwal

Goal: find relevant knowledge in ChromaDB, paste it into the prompt, and let the LLM answer.
We ask the SAME question twice (without and with RAG) so you can compare the answers.

Run it:
    python 05_rag_chain.py
"""

import chromadb
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# ----------------------------------------------------------------------------
# 1) Build the knowledge base (same idea as Step 03)
# ----------------------------------------------------------------------------
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

client = chromadb.Client()
col = client.create_collection("inv_kb", metadata={"hnsw:space": "cosine"})

KB = [
    ("ss",   "Safety Stock = Z * demand_std_dev * sqrt(lead_time). Z=1.65 for 95%, Z=2.33 for 99%."),
    ("rop",  "Reorder Point (ROP) = (avg_daily_demand * lead_time) + Safety Stock."),
    ("elec", "Electronics: lead times 14-30 days, high obsolescence risk."),
    ("ph",   "Pharma: 99%+ service level, strict FIFO, expiry dates matter."),
    ("slv",  "Raising service level from 95% to 99% increases safety stock by about 40% (Z goes from 1.65 to 2.33)."),
    ("var",  "High demand variability requires more safety stock for the same service level."),
]
vecs = embed.embed_documents([text for _, text in KB])
col.add(ids=[doc_id for doc_id, _ in KB], embeddings=vecs, documents=[text for _, text in KB])


def retrieve_context(query: str, n: int = 3) -> str:
    """RETRIEVE: return the n most relevant KB passages as one bullet-list string."""
    # ----- TODO 1: Search the collection and return the passages as a bullet list ("- passage" per line) -----
    # Hint: qv = embed.embed_query(query); res = col.query(query_embeddings=[qv], n_results=n)
    # Hint: The passages are in res["documents"][0]
    # Hint: "\n".join(f"- {c}" for c in passages)
    return ""  # replace with your code


# ----------------------------------------------------------------------------
# 2) Two prompts: one WITHOUT extra knowledge, one WITH a {rag_context} slot
# ----------------------------------------------------------------------------
human_msg = (
    "SKU: {sku_name} | Category: {category}\n"
    "Current Stock: {current_stock} | Lead Time: {lead_time_days} days\n"
    "Daily Demand: {avg_daily_demand} (std dev {demand_std_dev}) | Service Level: {service_level}%"
)

bare_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an inventory planning expert. Advise on reorder strategy in under 150 words."),
    ("human", human_msg),
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an inventory planning expert. Advise on reorder strategy in under 150 words.\n\n"
     "Use this domain knowledge:\n{rag_context}"),   # AUGMENT: retrieved text goes here
    ("human", human_msg),
])

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")
bare_chain = bare_prompt | llm | StrOutputParser()
rag_chain = rag_prompt | llm | StrOutputParser()

# ----------------------------------------------------------------------------
# 3) Compare
# ----------------------------------------------------------------------------
sku = {
    "sku_name": "Paracetamol 500mg",
    "category": "Pharma",
    "current_stock": 800,
    "lead_time_days": 7,
    "avg_daily_demand": 40,
    "demand_std_dev": 5,
    "service_level": 99,
}

query = f"{sku['category']} inventory reorder safety stock {sku['service_level']}%"
context = retrieve_context(query)

print("=== Retrieved Context ===")
print(context)

print("\n=== Response WITHOUT RAG ===")
print(bare_chain.invoke(sku))

print("\n=== Response WITH RAG ===")
# ----- TODO 2: Run the RAG chain: pass all SKU fields AND the retrieved context -----
# Hint: rag_chain.invoke({**sku, "rag_context": context})   ({**sku} copies every key of the dict)
answer = "TODO"  # replace with your code
print(answer)
