"""
trip_04_llm_chain.py
────────────────────
Lesson 04 · LLM chain: ask OpenAI to work out the cost of a trip.

The story
    We hire an AI "trip planner". We write the cost rules into its instructions
    (the system prompt) and send each trip's numbers in the request. It works out
    how much each friend has to pay.

What you learn
    - ChatPromptTemplate: a reusable prompt with {placeholders}
    - LCEL pipe syntax: prompt | llm | parser (data flows left to right)
    - System message = the AI's job and rules; human message = this request's data
    - No retrieval yet: the cost rules are written directly into the prompt

    Watch the maths! Here the LLM does the arithmetic itself (rounding rooms up,
    dividing by 3...). It is usually close but not always exact.
    Lesson 06 fixes that with tools.

How to run
    1. cp .env.example .env   and paste your OPENAI_API_KEY
    2. uv run trip_04_llm_chain.py

What you will see
    Step-by-step working for 2 trips, each ending with COST_PER_PERSON: <number>.
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()   # reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env

# ── The model ─────────────────────────────────────────────────────────────────
# gpt-6-luna is OpenAI's low-cost model. It can "think" before answering;
# reasoning_effort="none" turns that off so replies are quick, like a classic
# chat model. To try another model, set OPENAI_MODEL in .env.
# (Older non-reasoning models such as gpt-4.1-mini: remove reasoning_effort
#  and use temperature=0 instead.)
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
llm = ChatOpenAI(model=MODEL, reasoning_effort="none")

# ── The prompt template ───────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a trip planner for a group of friends.

Given these numbers, compute the cost per person step by step.

Rules:
  Rooms needed     = people / 2, rounded UP (one room fits 2 people)
  Stay share       = rooms needed * room price per night * nights / people
  Food and fun     = daily spend per person * nights
  Cost per person  = travel cost + stay share + food and fun

End your response with exactly:
COST_PER_PERSON: <number>"""),
    ("human",
     "Place: {place}\n"
     "Friends: {people}\n"
     "Nights: {nights}\n"
     "Room price per night: ₹{room_price_per_night}\n"
     "Travel cost per person (return): ₹{travel_cost_per_person}\n"
     "Daily spend per person: ₹{daily_spend_per_person}"),
])

# ── The chain: prompt → model → plain string ──────────────────────────────────
chain = prompt | llm | StrOutputParser()

test_cases = [
    {"place": "Goa",    "people": 4, "nights": 3, "room_price_per_night": 3000, "travel_cost_per_person": 2800, "daily_spend_per_person": 1500},
    {"place": "Manali", "people": 3, "nights": 4, "room_price_per_night": 2500, "travel_cost_per_person": 9000, "daily_spend_per_person": 1300},
]
# Correct answers (from the tool in lesson 06), so you can check the LLM:
#   Goa → 11800        Manali → 20866.67  (3 friends still need 2 rooms!)

for params in test_cases:
    print(f"\n{'='*60}")
    print(f"Trip: {params['place']} — {params['people']} friends, {params['nights']} nights")
    print("="*60)
    result = chain.invoke(params)   # fills the {placeholders}, calls OpenAI, returns text
    print(result)
