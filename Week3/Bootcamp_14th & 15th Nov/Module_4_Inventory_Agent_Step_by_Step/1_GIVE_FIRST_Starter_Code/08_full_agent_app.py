"""
Step 08 — The full Inventory Planning Agent (DB + RAG + Tools + Chat UI)
Created by Shivank Agarwal

Everything from Steps 01-07 in one app:
  01 SQLite          -> 3 tables, 10 products
  02/03 Embeddings   -> ChromaDB knowledge base (13 short notes)
  05 RAG             -> relevant notes are added to the system prompt for every question
  06 Tools + loop    -> the agent looks up data and does exact maths with tools
  NEW                -> a Gradio chat window anyone can use in the browser

Run it:
    python 08_full_agent_app.py
Then open the link printed in the terminal (usually http://127.0.0.1:7860).
"""

import json
import math
import sqlite3

import chromadb
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# =============================================================================
# 1. Database (3 tables, 10 SKUs)
# =============================================================================
# check_same_thread=False: Gradio answers in a different thread than the one that created the DB
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.executescript("""
CREATE TABLE inventory (
    sku_id TEXT PRIMARY KEY, sku_name TEXT NOT NULL, category TEXT,
    current_stock REAL NOT NULL, service_level REAL NOT NULL
);
CREATE TABLE lead_time (
    sku_id TEXT PRIMARY KEY, lead_time_days REAL NOT NULL, supplier_name TEXT,
    FOREIGN KEY (sku_id) REFERENCES inventory(sku_id)
);
CREATE TABLE demand (
    sku_id TEXT PRIMARY KEY, avg_daily_demand REAL NOT NULL, demand_std_dev REAL NOT NULL,
    FOREIGN KEY (sku_id) REFERENCES inventory(sku_id)
);
""")

cur.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", [
    ("SKU-001", "Wireless Headphones",     "Electronics",   150, 98),
    ("SKU-002", "Paracetamol 500mg",       "Pharma",        800, 99),
    ("SKU-003", "Instant Noodles 12-pack", "FMCG",          300, 95),
    ("SKU-004", "Steel Rod 10mm",          "Raw Materials", 300, 92),
    ("SKU-005", "Laptop 15-inch",          "Electronics",    50, 99),
    ("SKU-006", "Vitamin C Tablets",       "Pharma",        600, 97),
    ("SKU-007", "Shampoo 400ml",           "FMCG",          900, 95),
    ("SKU-008", "Copper Wire 2.5mm",       "Raw Materials", 150, 90),
    ("SKU-009", "Bluetooth Speaker",       "Electronics",   120, 96),
    ("SKU-010", "Flour 5kg",               "FMCG",         2000, 95),
])
cur.executemany("INSERT INTO lead_time VALUES (?,?,?)", [
    ("SKU-001", 21, "TechImports Ltd"), ("SKU-002", 7, "MedSupply Co"),
    ("SKU-003", 4, "FoodDist Pvt"),     ("SKU-004", 14, "SteelWorks Inc"),
    ("SKU-005", 30, "GlobalTech"),      ("SKU-006", 10, "PharmaPlus"),
    ("SKU-007", 5, "FMCG Direct"),      ("SKU-008", 21, "CopperMine Co"),
    ("SKU-009", 18, "AudioWorld"),      ("SKU-010", 3, "LocalMills"),
])
cur.executemany("INSERT INTO demand VALUES (?,?,?)", [
    ("SKU-001", 8, 2.0),  ("SKU-002", 40, 5.0), ("SKU-003", 80, 10.0),
    ("SKU-004", 15, 3.0), ("SKU-005", 3, 1.0),  ("SKU-006", 25, 4.0),
    ("SKU-007", 60, 8.0), ("SKU-008", 10, 2.5), ("SKU-009", 6, 1.5),
    ("SKU-010", 150, 20.0),
])
conn.commit()


def fetch_sku(sku_id: str) -> dict | None:
    row = cur.execute("""
        SELECT i.sku_id, i.sku_name, i.category, i.current_stock, i.service_level,
               l.lead_time_days, l.supplier_name,
               d.avg_daily_demand, d.demand_std_dev
        FROM inventory i
        JOIN lead_time l ON l.sku_id = i.sku_id
        JOIN demand    d ON d.sku_id = i.sku_id
        WHERE UPPER(i.sku_id) = UPPER(?)
    """, (sku_id,)).fetchone()
    return dict(row) if row else None


def list_skus() -> list[dict]:
    rows = cur.execute("SELECT sku_id, sku_name, category, current_stock FROM inventory ORDER BY sku_id")
    return [dict(r) for r in rows.fetchall()]


# =============================================================================
# 2. RAG knowledge base (ChromaDB)
# =============================================================================
KB = [
    ("ss-formula",    "Safety Stock = Z * demand_std_dev * sqrt(lead_time). Z=1.28 for 90%, 1.65 for 95%, 2.05 for 98%, 2.33 for 99%.", {"topic": "formula"}),
    ("rop-formula",   "Reorder Point (ROP) = (avg_daily_demand * lead_time) + safety_stock.",                                        {"topic": "formula"}),
    ("desired",       "Desired Stock = (avg_daily_demand * lead_time) + safety_stock. Reorder qty = max(0, Desired - Current).",     {"topic": "formula"}),
    ("eoq",           "EOQ (Economic Order Quantity) = sqrt((2 * Annual Demand * Ordering Cost) / Holding Cost).",                   {"topic": "formula"}),
    ("electronics",   "Electronics: lead times 14-30 days, service level 95-99%, high obsolescence risk.",                           {"topic": "category"}),
    ("fmcg",          "FMCG: lead times 3-7 days, high steady demand, service level 95-98%.",                                        {"topic": "category"}),
    ("pharma",        "Pharma: lead times 7-21 days, 99%+ service level, strict FIFO and expiry management.",                        {"topic": "category"}),
    ("raw-mat",       "Raw materials: lead times 10-45 days, service level 90-95%.",                                                 {"topic": "category"}),
    ("abc",           "ABC Analysis: A-items need tight control; C-items tolerate bulk ordering.",                                    {"topic": "best_practice"}),
    ("variability",   "High demand variability requires more safety stock for the same service level.",                               {"topic": "best_practice"}),
    ("seasonal",      "Increase safety stock 4-6 weeks before peak season (e.g. Diwali); reduce quickly after the season.",          {"topic": "best_practice"}),
    ("supplier-risk", "Single-source or long international lead times: add a 10-20% risk buffer to safety stock.",                    {"topic": "best_practice"}),
    ("sl-tradeoff",   "Raising service level from 95% to 99% increases safety stock by about 40% (Z goes from 1.65 to 2.33).",       {"topic": "best_practice"}),
]

embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

chroma = chromadb.Client()
col = chroma.create_collection("inventory_kb", metadata={"hnsw:space": "cosine"})
col.add(
    ids=[doc_id for doc_id, _, _ in KB],
    embeddings=embed.embed_documents([text for _, text, _ in KB]),
    documents=[text for _, text, _ in KB],
    metadatas=[meta for _, _, meta in KB],
)


def retrieve_context(query: str, n: int = 3) -> str:
    qv = embed.embed_query(query)
    res = col.query(query_embeddings=[qv], n_results=n)
    return "\n".join(f"- {c}" for c in res["documents"][0])


# =============================================================================
# 3. Tools
# =============================================================================
Z_TABLE = {90: 1.28, 91: 1.34, 92: 1.41, 93: 1.48, 94: 1.55, 95: 1.65, 96: 1.75, 97: 1.88, 98: 2.05, 99: 2.33}


@tool
def db_lookup(sku_id: str) -> dict:
    """Fetch all inventory, lead-time, and demand parameters for a SKU from the database."""
    record = fetch_sku(sku_id)
    return record if record else {"error": f"{sku_id} not found. Use list_available_skus to see valid IDs."}


@tool
def list_available_skus() -> list[dict]:
    """Return all SKUs currently in the inventory database."""
    return list_skus()


# ----- TODO 1: Write the compute_stock_metrics tool (exact maths for one SKU) -----
# Hint: Inputs: avg_daily_demand, demand_std_dev, lead_time_days, service_level_pct, current_stock (all float)
# Hint: z = Z_TABLE.get(round(service_level_pct), 1.65)
# Hint: safety_stock = Z * demand_std_dev * sqrt(lead_time_days);  desired_stock = avg_daily_demand * lead_time_days + safety_stock
# Hint: reorder_quantity = max(0, math.ceil(desired_stock - current_stock))   (whole units)
# Hint: Return a dict with z_score, safety_stock, desired_stock, reorder_quantity, reorder_needed
@tool
def compute_stock_metrics(avg_daily_demand: float, demand_std_dev: float, lead_time_days: float,
                          service_level_pct: float, current_stock: float) -> dict:
    """Compute safety stock, desired stock level, and reorder quantity with exact arithmetic."""
    return {}  # replace with your code


TOOLS = [db_lookup, list_available_skus, compute_stock_metrics]
TOOL_MAP = {t.name: t for t in TOOLS}

# =============================================================================
# 4. The agent
# =============================================================================
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none").bind_tools(TOOLS)

SYSTEM = """You are an inventory planning agent with access to tools and domain knowledge.

When a user asks about a SKU:
1. Use db_lookup to fetch its parameters
2. Use compute_stock_metrics for exact arithmetic (never do the maths yourself)
3. Give a clear recommendation: reorder quantity, urgency, and a short justification

When the user asks to list SKUs, call list_available_skus.
When a question is about several SKUs, look up each one.

Domain knowledge (use this for context and best practices):
{rag_context}"""

MAX_STEPS = 15  # a question about all 10 SKUs needs many tool rounds


def run_agent(user_message: str) -> str:
    # RAG: find notes related to the question and put them in the system prompt
    rag_context = retrieve_context(user_message)
    messages = [
        SystemMessage(content=SYSTEM.format(rag_context=rag_context)),
        HumanMessage(content=user_message),
    ]

    # Agent loop: think -> call tools -> look at results -> think again ...
    for _ in range(MAX_STEPS):
        # ----- TODO 2: One turn of the agent loop (same as Step 06) -----
        # Hint: response = llm.invoke(messages); messages.append(response)
        # Hint: No tool_calls -> return response.text (the final answer as plain text)
        # Hint: Otherwise run each tool and append ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc["id"])
        return "TODO"  # replace with your code

    return "Sorry, I could not finish this request. Please ask about fewer SKUs at a time."


# =============================================================================
# 5. Gradio chat window
# =============================================================================
def chat_handler(message, history):
    # history = earlier messages in the chat window (not used: every question is answered on its own)
    try:
        return run_agent(message)
    except Exception as e:
        return f"Something went wrong: {e}\n\nTry: *What is the reorder quantity for SKU-001?* or type **list**."


demo = gr.ChatInterface(
    fn=chat_handler,
    title="Inventory Planning Agent (DB + RAG + Tools)",
    description=(
        "Created by Shivank Agarwal. Ask about any SKU: the agent reads the database, "
        "does exact maths with tools, and uses domain knowledge for advice. Type **list** to see all SKUs."
    ),
    examples=[
        "list",
        "What is the reorder quantity for SKU-001?",
        "Should I reorder SKU-002? It's a pharma product.",
        "Give me a full analysis for SKU-005.",
        "How much should I order for SKU-010?",
        "Which SKUs are most at risk of stockout?",
    ],
)

if __name__ == "__main__":
    print("Starting Inventory Planning Agent...")
    print(f"  {len(list_skus())} SKUs in the database")
    print(f"  {col.count()} notes in the knowledge base")
    demo.launch()
