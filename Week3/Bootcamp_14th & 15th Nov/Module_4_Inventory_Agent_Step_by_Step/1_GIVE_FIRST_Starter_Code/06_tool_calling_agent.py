"""
Step 06 — Tool calling with an agent LOOP
Created by Shivank Agarwal

In Step 04 the LLM did the maths itself (and can get it wrong).
Now the maths is done by Python tools, and the LLM only decides WHICH tool to call and WHEN.

New idea: a LOOP. Some questions need several tools one after another, e.g.
   "Does SKU-005 need to be reordered?"
     -> lookup_sku_stock        (get the numbers from the "database")
     -> calculate_safety_stock  (exact maths)
     -> calculate_reorder_point (exact maths)
     -> final answer
We keep calling the LLM until it stops asking for tools.

Run it:
    python 06_tool_calling_agent.py
"""

import json
import math

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# Z-score for each service level (standard normal table)
Z_TABLE = {90: 1.28, 91: 1.34, 92: 1.41, 93: 1.48, 94: 1.55, 95: 1.65, 96: 1.75, 97: 1.88, 98: 2.05, 99: 2.33}


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def calculate_safety_stock(demand_std_dev: float, lead_time_days: float, service_level_pct: float) -> float:
    """
    Calculate safety stock (extra buffer units).
    safety_stock = Z * demand_std_dev * sqrt(lead_time_days)
    Z-scores: 90%->1.28, 95%->1.65, 97%->1.88, 98%->2.05, 99%->2.33
    """
    z = Z_TABLE.get(round(service_level_pct), 1.65)
    return round(z * demand_std_dev * math.sqrt(lead_time_days), 2)


@tool
def calculate_reorder_point(avg_daily_demand: float, lead_time_days: float, safety_stock: float) -> float:
    """
    Calculate the reorder point (ROP): reorder when stock falls to this level.
    ROP = (avg_daily_demand * lead_time_days) + safety_stock
    """
    return round(avg_daily_demand * lead_time_days + safety_stock, 2)


@tool
def lookup_sku_stock(sku_id: str) -> dict:
    """
    Look up current stock and demand parameters for a SKU from the inventory database.
    Returns: sku_id, sku_name, current_stock, avg_daily_demand, demand_std_dev, lead_time_days, service_level
    """
    db = {  # a small pretend database
        "SKU-001": {"sku_name": "Wireless Headphones", "current_stock": 150, "avg_daily_demand": 8,  "demand_std_dev": 2.0, "lead_time_days": 21, "service_level": 98},
        "SKU-002": {"sku_name": "Paracetamol 500mg",   "current_stock": 800, "avg_daily_demand": 40, "demand_std_dev": 5.0, "lead_time_days": 7,  "service_level": 99},
        "SKU-005": {"sku_name": "Laptop 15-inch",      "current_stock": 50,  "avg_daily_demand": 3,  "demand_std_dev": 1.0, "lead_time_days": 30, "service_level": 99},
    }
    record = db.get(sku_id.upper())
    if not record:
        return {"error": f"{sku_id} not found"}
    return {"sku_id": sku_id.upper(), **record}


# ── LLM with tools bound ──────────────────────────────────────────────────────
tools = [calculate_safety_stock, calculate_reorder_point, lookup_sku_stock]
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none").bind_tools(tools)
TOOL_MAP = {t.name: t for t in tools}

MAX_STEPS = 8  # safety net: never loop forever


# ── The agent loop ────────────────────────────────────────────────────────────
def run_agent(user_question: str) -> str:
    messages = [HumanMessage(content=user_question)]

    for _ in range(MAX_STEPS):
        # ----- TODO 1: One turn of the agent loop -----
        # Hint: response = llm.invoke(messages); messages.append(response)
        # Hint: If response.tool_calls is empty -> return response.text (the final answer as plain text)
        # Hint: Otherwise run each tool: TOOL_MAP[tc["name"]].invoke(tc["args"])
        # Hint: and append ToolMessage(content=json.dumps(result), tool_call_id=tc["id"])
        return "TODO"  # replace with your code

    return "Stopped: too many steps without a final answer."


if __name__ == "__main__":
    questions = [
        "What is the safety stock for SKU-001?",
        "Does SKU-005 need to be reordered? Show the reorder point.",
        "Calculate the reorder point for an item with daily demand of 60, demand std dev of 8, lead time 5 days, and 95% service level.",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {run_agent(q)}")
