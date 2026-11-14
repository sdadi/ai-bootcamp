"""
Exercise 3.3 — An LLM that can use FOUR tools (warehouse assistant)
Created by Shivank Agarwal

Same pattern as 3.2, but now the LLM must CHOOSE the right tool from a menu of four,
and fill in the arguments (SKU id, lead time) from the user's plain-English question.

Run it:
    python 3_inventory_tools_demo.py
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ----------------------------------------------------------------------------
# A tiny "database" (a Python dict) of warehouse items. SKU = Stock Keeping Unit = product id.
# reorder_point = when stock falls to this level or below, we should order more.
# ----------------------------------------------------------------------------
INVENTORY = {
    "SKU-001": {"name": "Industrial Valve", "stock": 120, "reorder_point": 50,  "unit_cost": 7000},
    "SKU-002": {"name": "Hydraulic Pump",   "stock": 18,  "reorder_point": 25,  "unit_cost": 35000},
    "SKU-003": {"name": "Conveyor Belt",    "stock": 5,   "reorder_point": 10,  "unit_cost": 17500},
    "SKU-004": {"name": "Safety Helmet",    "stock": 300, "reorder_point": 100, "unit_cost": 1000},
    "SKU-005": {"name": "Pressure Gauge",   "stock": 42,  "reorder_point": 40,  "unit_cost": 4500},
}

# How many units of each item we use per day (on average)
DAILY_DEMAND = {"SKU-001": 8, "SKU-002": 3, "SKU-003": 2, "SKU-004": 15, "SKU-005": 6}


# ----------------------------------------------------------------------------
# Tools. The docstring is what the LLM reads to decide WHICH tool to use, so it must be clear.
# ----------------------------------------------------------------------------
@tool
def get_stock_level(sku: str) -> str:
    """Return current stock level and reorder point for a given SKU."""
    sku = sku.upper()
    item = INVENTORY.get(sku)
    if not item:
        return f"SKU {sku} not found."
    status = "LOW" if item["stock"] <= item["reorder_point"] else "OK"
    return f"{item['name']} ({sku}): stock={item['stock']}, reorder_point={item['reorder_point']}, status={status}"


@tool
def calculate_reorder_quantity(sku: str, lead_time_days: int = 7) -> str:
    """Calculate recommended reorder quantity for a SKU given supplier lead time in days."""
    sku = sku.upper()
    item = INVENTORY.get(sku)
    if not item:
        return f"SKU {sku} not found."
    daily = DAILY_DEMAND.get(sku, 1)
    # Simple rule of thumb:
    #   demand during delivery wait  = daily demand x lead time
    #   safety buffer                = the same amount again (in case of delays / spikes)
    #   order                        = (wait demand + buffer) - what we already have
    lead_time_demand = daily * lead_time_days
    safety_buffer = lead_time_demand
    reorder_qty = max(lead_time_demand + safety_buffer - item["stock"], 0)
    total_cost = reorder_qty * item["unit_cost"]
    return (
        f"{item['name']} ({sku}): recommend ordering {reorder_qty} units "
        f"(lead_time_demand={lead_time_demand}, safety_buffer={safety_buffer}, "
        f"lead_time={lead_time_days}d, estimated_cost=Rs {total_cost:,})"
    )


@tool
def days_until_stockout(sku: str) -> str:
    """Estimate how many days until a SKU runs out of stock at current demand rate."""
    sku = sku.upper()
    item = INVENTORY.get(sku)
    if not item:
        return f"SKU {sku} not found."
    daily = DAILY_DEMAND.get(sku, 1)
    days = item["stock"] / daily
    return f"{item['name']} ({sku}): {days:.1f} days until stockout (stock={item['stock']}, daily_demand={daily})"


# ----- TODO 1: Write a 4th tool that lists every SKU at or below its reorder point -----
# Hint: Decorate with @tool and write a clear docstring (the LLM reads it!)
# Hint: Loop over INVENTORY.items() and keep items where stock <= reorder_point
# Hint: Return a string, e.g. "Items needing reorder (2 total): SKU-002 (...); SKU-003 (...)"
@tool
def list_low_stock_items() -> str:
    """List all SKUs that are at or below their reorder point and need restocking."""
    return "TODO"  # replace with your code


# ----- TODO 2: Put all four tools in a list, build tool_map, and bind the tools to the LLM -----
# Hint: tool_map = {t.name: t for t in tools}
# Hint: ChatOpenAI(model="gpt-6-luna", reasoning_effort="none").bind_tools(tools)
tools = []
tool_map = {}
llm_with_tools = None  # replace with your code


def chat_with_tools(user_message: str) -> str:
    """Same 4-step loop as Exercise 3.2."""
    messages = [HumanMessage(content=user_message)]
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        return response.text  # .text = the answer as plain text

    for tc in response.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        print(f"   [tool] {tc['name']}({tc['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return llm_with_tools.invoke(messages).text


if __name__ == "__main__":
    questions = [
        "What is the current stock level for SKU-003?",
        "Which items need to be reordered right now?",
        "How many days until SKU-002 runs out of stock?",
        "What quantity of SKU-003 should I order if my supplier takes 10 days to deliver?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {chat_with_tools(q)}")
