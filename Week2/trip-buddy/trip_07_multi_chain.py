"""
trip_07_multi_chain.py
──────────────────────
Lesson 07 · Multi-chain pipeline: compute first, then decide.

The story
    Our trip options now live in a real 3-table database. For each trip plan we
    run a fixed assembly line:
      1. Python + tool  → exact cost per person (no AI guessing)
      2. Chain 1 (AI)   → explains the cost breakdown in plain English
      3. Chain 2 (AI)   → reads Chain 1's explanation and says "Pack your bags!"
                          or "Over budget, here is what to change"
    The output of one chain becomes the input of the next.

What you learn
    - 3 tables (places, stays, costs) joined into one record
    - Calling a @tool directly from Python with .invoke({...}) (no LLM needed to call it)
    - Chaining LLM steps: chain 2 consumes chain 1's text
    - Telling the LLM "these numbers come from a trusted tool, do NOT redo them"

How to run
    uv run trip_07_multi_chain.py        (needs OPENAI_API_KEY in .env)

What you will see
    For Goa and Manali: the tool result, Chain 1's breakdown, Chain 2's decision.
"""

import math
import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 1. Database: 3 tables, 3 places ───────────────────────────────────────────
conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.executescript("""
CREATE TABLE places (
    place_id TEXT PRIMARY KEY, name TEXT, trip_type TEXT, best_months TEXT
);
CREATE TABLE stays (
    place_id TEXT PRIMARY KEY, stay_name TEXT, room_price_per_night REAL
);
CREATE TABLE costs (
    place_id TEXT PRIMARY KEY, travel_mode TEXT,
    travel_cost_per_person REAL, daily_spend_per_person REAL
);
""")
cur.executemany("INSERT INTO places VALUES (?,?,?,?)", [
    ("GOA",    "Goa",    "Beach", "Nov–Feb"),
    ("COORG",  "Coorg",  "Hills", "Oct–Jun"),
    ("MANALI", "Manali", "Hills", "Mar–Jun"),
])
cur.executemany("INSERT INTO stays VALUES (?,?,?)", [
    ("GOA",    "Sea Breeze Guesthouse", 3000),
    ("COORG",  "Coffee Estate Cottage", 2800),
    ("MANALI", "Old Manali Cafe Stay",  2500),
])
cur.executemany("INSERT INTO costs VALUES (?,?,?,?)", [
    ("GOA",    "Overnight train", 2800, 1500),
    ("COORG",  "Bus",             1000, 1100),
    ("MANALI", "Flight + bus",    9000, 1300),
])
conn.commit()


def fetch_place(place: str) -> dict | None:
    """JOIN all 3 tables and return one place's full record as a dict."""
    row = cur.execute("""
        SELECT p.*, s.stay_name, s.room_price_per_night,
               c.travel_mode, c.travel_cost_per_person, c.daily_spend_per_person
        FROM places p
        JOIN stays s ON s.place_id = p.place_id
        JOIN costs c ON c.place_id = p.place_id
        WHERE UPPER(p.place_id) = UPPER(?) OR UPPER(p.name) = UPPER(?)
    """, (place, place)).fetchone()
    return dict(row) if row else None


# ── 2. The calculator tool ────────────────────────────────────────────────────
@tool
def trip_cost_tool(travel_cost_per_person: float, room_price_per_night: float,
                   daily_spend_per_person: float, people: int, nights: int) -> dict:
    """Compute rooms needed, stay share, food & fun, and total cost per person with exact arithmetic."""
    rooms_needed = math.ceil(people / 2)
    stay_share = round(rooms_needed * room_price_per_night * nights / people, 2)
    food_and_fun = round(daily_spend_per_person * nights, 2)
    total = round(travel_cost_per_person + stay_share + food_and_fun, 2)
    return {"rooms_needed": rooms_needed, "stay_share": stay_share,
            "food_and_fun": food_and_fun, "total_per_person": total}


MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
llm = ChatOpenAI(model=MODEL, reasoning_effort="none")   # see lesson 04 for why

# ── 3. Chain 1: explain the cost breakdown ────────────────────────────────────
explain_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a trip planner for a group of friends.\n"
     "The numbers below were calculated by a trusted tool — do NOT redo them.\n"
     "Explain the cost breakdown in simple words and end with:\nCOST_PER_PERSON: {total_per_person}"),
    ("human",
     "Trip: {name} ({trip_type}), best months {best_months}\n"
     "Group: {people} friends, {nights} nights at {stay_name}\n"
     "Travel: {travel_mode}, ₹{travel_cost_per_person} per person\n\n"
     "Tool results:\n"
     "  Rooms needed     : {rooms_needed}\n"
     "  Stay share       : ₹{stay_share}\n"
     "  Food and fun     : ₹{food_and_fun}\n"
     "  Total per person : ₹{total_per_person}"),
])
explain_chain = explain_prompt | llm | StrOutputParser()

# ── 4. Chain 2: go / no-go decision (reads chain 1's output) ──────────────────
decide_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the group's money-minded friend.\n"
     "Money left per person = budget - total cost per person.\n"
     "Say clearly: GO (within budget) or NOT YET (over budget). "
     "If over budget, suggest one simple change (fewer nights, cheaper place, more friends)."),
    ("human",
     "Trip: {name} | Budget per person: ₹{budget_per_person}\n\n"
     "Cost breakdown:\n{breakdown}"),
])
decide_chain = decide_prompt | llm | StrOutputParser()


def run_pipeline(place: str, people: int, nights: int, budget_per_person: float) -> None:
    """DB → tool → chain 1 → chain 2, printing each stage."""
    record = fetch_place(place)
    if not record:
        print(f"Place {place} not found.")
        return

    # Step 1: exact arithmetic via the tool (Python calls it directly, no LLM)
    tool_result = trip_cost_tool.invoke({
        "travel_cost_per_person": record["travel_cost_per_person"],
        "room_price_per_night":   record["room_price_per_night"],
        "daily_spend_per_person": record["daily_spend_per_person"],
        "people":                 people,
        "nights":                 nights,
    })
    print(f"\n[Tool] trip_cost_tool → {tool_result}")

    # Step 2: chain 1 explains the numbers ({**a, **b} merges dicts)
    breakdown = explain_chain.invoke({**record, **tool_result, "people": people, "nights": nights})
    print(f"\n--- Chain 1: Cost Breakdown ---\n{breakdown}")

    # Step 3: chain 2 turns that explanation into a decision
    decision = decide_chain.invoke({
        "name":              record["name"],
        "budget_per_person": budget_per_person,
        "breakdown":         breakdown,      # ← chain 1's output is chain 2's input
    })
    print(f"\n--- Chain 2: Go or Not Yet ---\n{decision}")


# ── Run ───────────────────────────────────────────────────────────────────────
plans = [
    ("Goa",    4, 3, 12000),   # 4 friends, 3 nights, ₹12,000 each
    ("Manali", 3, 4, 20000),   # 3 friends, 4 nights, ₹20,000 each
]
for place, people, nights, budget in plans:
    print(f"\n{'='*60}")
    print(f"Plan: {place} — {people} friends, {nights} nights, ₹{budget} each")
    print("="*60)
    run_pipeline(place, people, nights, budget)
