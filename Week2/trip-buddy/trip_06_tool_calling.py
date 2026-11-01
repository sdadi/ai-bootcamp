"""
trip_06_tool_calling.py
───────────────────────
Lesson 06 · Tool calling: let the AI use a calculator and a lookup instead of guessing.

The story
    In lesson 04 the AI did the maths in its head and sometimes slipped.
    Now we give it three tools, like a calculator and a travel diary on the table:
      - lookup_place          → read a place's prices
      - calculate_stay_share  → exact "how much does each friend pay for rooms"
      - calculate_trip_cost   → exact total per person
    The AI decides WHICH tool to call and with WHAT inputs. Python runs the tool
    and hands the result back. The AI only writes the final explanation.

What you learn
    - @tool turns a normal Python function into something the LLM can call
      (the docstring and type hints tell the LLM what the tool does)
    - llm.bind_tools(tools) shows the tool menu to the model
    - The tool loop: model asks → we run the tool → send ToolMessage back → repeat

How to run
    uv run trip_06_tool_calling.py        (needs OPENAI_API_KEY in .env)

What you will see
    For each question, every tool call the AI made, then its final answer.
"""

import json
import math
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def calculate_stay_share(room_price_per_night: float, nights: int, people: int) -> dict:
    """
    Calculate how much each person pays for the stay.
    One room fits 2 people, so rooms_needed = people / 2 rounded up.
    stay_share = rooms_needed * room_price_per_night * nights / people
    """
    rooms_needed = math.ceil(people / 2)
    stay_share = round(rooms_needed * room_price_per_night * nights / people, 2)
    return {"rooms_needed": rooms_needed, "stay_share": stay_share}


@tool
def calculate_trip_cost(travel_cost_per_person: float, stay_share: float,
                        daily_spend_per_person: float, nights: int) -> float:
    """
    Calculate the total trip cost for one person.
    total = travel_cost_per_person + stay_share + daily_spend_per_person * nights
    """
    return round(travel_cost_per_person + stay_share + daily_spend_per_person * nights, 2)


@tool
def lookup_place(place: str) -> dict:
    """
    Look up a destination's prices.
    Returns: place, trip_type, room_price_per_night, travel_cost_per_person, daily_spend_per_person
    """
    # Simulated database (a real one arrives in lesson 07)
    db = {
        "GOA":    {"trip_type": "Beach", "room_price_per_night": 3000, "travel_cost_per_person": 2800, "daily_spend_per_person": 1500},
        "COORG":  {"trip_type": "Hills", "room_price_per_night": 2800, "travel_cost_per_person": 1000, "daily_spend_per_person": 1100},
        "MANALI": {"trip_type": "Hills", "room_price_per_night": 2500, "travel_cost_per_person": 9000, "daily_spend_per_person": 1300},
    }
    record = db.get(place.upper())
    if not record:
        return {"error": f"{place} not found"}
    return {"place": place.title(), **record}


# ── LLM with tools bound ──────────────────────────────────────────────────────
tools = [calculate_stay_share, calculate_trip_cost, lookup_place]

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
llm = ChatOpenAI(model=MODEL, reasoning_effort="none").bind_tools(tools)

# Map tool name → tool, so we can run whichever one the model asks for
TOOL_MAP = {t.name: t for t in tools}

MAX_STEPS = 8   # safety net: stop if the model keeps calling tools forever


# ── Simple agent loop ─────────────────────────────────────────────────────────
def run_agent(user_question: str) -> str:
    """Run the tool-calling loop until the model gives a final answer."""
    messages = [HumanMessage(content=user_question)]

    for _ in range(MAX_STEPS):
        response = llm.invoke(messages)   # model reads everything so far
        messages.append(response)

        # No tool calls → the model is done, this is the final answer
        if not response.tool_calls:
            return response.text

        # Otherwise run every tool the model asked for and send results back
        for tc in response.tool_calls:
            fn = TOOL_MAP[tc["name"]]
            result = fn.invoke(tc["args"])
            print(f"  [tool] {tc['name']}({tc['args']}) → {result}")
            messages.append(ToolMessage(
                content=json.dumps(result),
                tool_call_id=tc["id"],    # tells the model which call this answers
            ))

    return "Stopped: too many tool calls without a final answer."


# ── Try it ────────────────────────────────────────────────────────────────────
questions = [
    "We are 4 friends going to Goa for 3 nights. How much will the stay cost each of us?",
    "Can 3 friends do Manali for 4 nights with ₹20,000 each? Show the total per person.",
    "Calculate the trip cost per person: travel ₹1,200, stay share ₹2,400, daily spend ₹900 for 2 nights.",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = run_agent(q)
    print(f"A: {answer}")
