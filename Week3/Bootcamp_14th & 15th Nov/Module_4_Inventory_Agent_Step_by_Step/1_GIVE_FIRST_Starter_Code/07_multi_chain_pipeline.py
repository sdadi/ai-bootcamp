"""
Step 07 — Multi-chain pipeline: database -> tool -> chain 1 -> chain 2
Created by Shivank Agarwal

Here WE (not the LLM) decide the order of steps. This is a fixed "pipeline":

   fetch_sku()            read the SKU from SQLite (3 tables joined)
   safety_stock_tool      exact maths in Python
   compute_chain  (LLM)   explains the calculation
   reorder_chain  (LLM)   reads chain 1's output and recommends what to order

Pipeline = fixed order, predictable.  Agent (Step 06) = LLM picks the order, flexible.

Run it:
    python 07_multi_chain_pipeline.py
"""

import math
import sqlite3

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 1) Database: 3 tables ─────────────────────────────────────────────────────
conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.executescript("""
CREATE TABLE inventory (sku_id TEXT PRIMARY KEY, sku_name TEXT, category TEXT, current_stock REAL, service_level REAL);
CREATE TABLE lead_time (sku_id TEXT PRIMARY KEY, lead_time_days REAL, supplier_name TEXT);
CREATE TABLE demand    (sku_id TEXT PRIMARY KEY, avg_daily_demand REAL, demand_std_dev REAL);
""")
cur.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", [
    ("SKU-001", "Wireless Headphones", "Electronics", 150, 98),
    ("SKU-002", "Paracetamol 500mg",   "Pharma",      800, 99),
    ("SKU-005", "Laptop 15-inch",      "Electronics",  50, 99),
])
cur.executemany("INSERT INTO lead_time VALUES (?,?,?)", [
    ("SKU-001", 21, "TechImports Ltd"),
    ("SKU-002",  7, "MedSupply Co"),
    ("SKU-005", 30, "GlobalTech"),
])
cur.executemany("INSERT INTO demand VALUES (?,?,?)", [
    ("SKU-001", 8, 2.0), ("SKU-002", 40, 5.0), ("SKU-005", 3, 1.0),
])
conn.commit()


def fetch_sku(sku_id: str) -> dict | None:
    row = cur.execute("""
        SELECT i.*, l.lead_time_days, l.supplier_name, d.avg_daily_demand, d.demand_std_dev
        FROM inventory i
        JOIN lead_time l ON l.sku_id = i.sku_id
        JOIN demand    d ON d.sku_id = i.sku_id
        WHERE UPPER(i.sku_id) = UPPER(?)
    """, (sku_id,)).fetchone()
    return dict(row) if row else None


# ── 2) Tool: exact maths ──────────────────────────────────────────────────────
Z_TABLE = {90: 1.28, 91: 1.34, 92: 1.41, 93: 1.48, 94: 1.55, 95: 1.65, 96: 1.75, 97: 1.88, 98: 2.05, 99: 2.33}


@tool
def safety_stock_tool(avg_daily_demand: float, demand_std_dev: float, lead_time_days: float, service_level_pct: float) -> dict:
    """Compute safety stock and desired stock level with exact arithmetic."""
    z = Z_TABLE.get(round(service_level_pct), 1.65)
    ss = round(z * demand_std_dev * math.sqrt(lead_time_days), 2)
    desired = round(avg_daily_demand * lead_time_days + ss, 2)
    return {"z_score": z, "safety_stock": ss, "desired_stock": desired}


# ── 3) Two chains ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")

compute_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an inventory planning expert.\n"
     "The calculations below were done by a trusted tool — do NOT redo them.\n"
     "Explain the result clearly in under 120 words and end with:\nDESIRED_STOCK: {desired_stock}"),
    ("human",
     "SKU: {sku_id} | {sku_name} ({category})\n"
     "Current Stock   : {current_stock}\n"
     "Lead Time       : {lead_time_days} days\n"
     "Daily Demand    : {avg_daily_demand} units/day (std dev {demand_std_dev})\n"
     "Service Level   : {service_level}%\n\n"
     "Tool results:\n"
     "  Z-score       : {z_score}\n"
     "  Safety Stock  : {safety_stock}\n"
     "  Desired Stock : {desired_stock}"),
])
compute_chain = compute_prompt | llm | StrOutputParser()

# ----- TODO 1: Build reorder_chain: a prompt that receives chain 1's explanation as {computation}, then llm, then a parser -----
# Hint: System message: "Reorder Quantity = max(0, Desired Stock - Current Stock). State the quantity and whether an immediate reorder is needed."
# Hint: Human message must contain {sku_id}, {sku_name}, {current_stock}, {supplier_name} and {computation}
# Hint: reorder_chain = reorder_prompt | llm | StrOutputParser()
reorder_chain = None  # replace with your code


# ── 4) The pipeline ───────────────────────────────────────────────────────────
def run_pipeline(sku_id: str) -> None:
    record = fetch_sku(sku_id)
    if not record:
        print(f"SKU {sku_id} not found.")
        return

    # Step 1: exact maths via the tool (we call it ourselves; no LLM decides here)
    tool_result = safety_stock_tool.invoke({
        "avg_daily_demand": record["avg_daily_demand"],
        "demand_std_dev": record["demand_std_dev"],
        "lead_time_days": record["lead_time_days"],
        "service_level_pct": record["service_level"],
    })
    print(f"\n[Tool] safety_stock_tool -> {tool_result}")

    # Step 2: chain 1 explains the numbers
    computation = compute_chain.invoke({**record, **tool_result})
    print(f"\n--- Chain 1: Stock Computation ---\n{computation}")

    # ----- TODO 2: Step 3 — run reorder_chain, feeding it chain 1's output, and print the recommendation -----
    # Hint: reorder_chain.invoke({"sku_id": ..., "sku_name": ..., "current_stock": ..., "supplier_name": ..., "computation": computation})
    # Hint: The first four values come from `record`
    recommendation = "TODO"  # replace with your code
    print(f"\n--- Chain 2: Reorder Recommendation ---\n{recommendation}")


if __name__ == "__main__":
    for sku in ["SKU-001", "SKU-005"]:
        print(f"\n{'=' * 60}\nProcessing {sku}\n{'=' * 60}")
        run_pipeline(sku)
