"""
trip_08_full_agent.py
─────────────────────
Lesson 08 · The full Trip Buddy agent: database + RAG + tools + ReAct loop + chat UI.

The story
    Everything comes together. Your friend group opens a chat window and asks
    "We are 4 friends with ₹12,000 each. Can we do Goa for 3 nights?"
    or "Which trips fit ₹8,000 each for 3 of us, 2 nights?"
    The agent:
      1. pulls the most relevant tips from the travel notebook (RAG, ChromaDB)
      2. decides on its own which tools to call (ReAct loop):
           db_lookup          → fetch a place's prices from SQLite
           list_destinations  → see every place we could go
           compute_trip_cost  → exact rooms, stay share, total per person, budget check
      3. writes a clear answer: can we afford it, what it costs, and travel tips

Brings together
    01  SQLite: full 3-table schema, 10 destinations
    02  Embeddings: sentence-transformers
    03  ChromaDB: 13-tip travel notebook
    04  LLM chains: prompt + model
    05  RAG: retrieved tips injected into the system prompt
    06  Tool calling: exact arithmetic and DB lookup as tools
    07  Multi-step reasoning: compute → decide

New in this file
    - A ReAct-style agent loop (Reason → Act with a tool → Observe the result → repeat)
    - A Gradio chat interface in your browser
    - Type "list" to see all destinations
    - All prices come from the DB; you only say where, how many friends, nights and budget

How to run
    uv run trip_08_full_agent.py        (needs OPENAI_API_KEY in .env)
    then open the link it prints (usually http://127.0.0.1:7860)

Try asking
    list
    We are 4 friends with ₹12,000 each. Can we do Goa for 3 nights?
    Is Manali a good idea in July?
    Which trips fit ₹8,000 each for 3 friends, 2 nights?
"""

import json
import math
import os
import sqlite3

import chromadb
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


load_dotenv()

# =============================================================================
# 1. Database (3 tables, 10 destinations)
# =============================================================================
# check_same_thread=False: Gradio answers chats on worker threads, so the
# connection must be usable from a thread other than the one that created it.
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.executescript("""
CREATE TABLE places (
    place_id TEXT PRIMARY KEY, name TEXT NOT NULL, trip_type TEXT,
    state TEXT, best_months TEXT
);
CREATE TABLE stays (
    place_id TEXT PRIMARY KEY, stay_name TEXT, room_price_per_night REAL NOT NULL,
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);
CREATE TABLE costs (
    place_id TEXT PRIMARY KEY, travel_mode TEXT,
    travel_cost_per_person REAL NOT NULL, daily_spend_per_person REAL NOT NULL,
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);
""")

# 4 kinds of trips: Beach, Hills, Heritage, Adventure. Prices are from Bengaluru.
cur.executemany("INSERT INTO places VALUES (?,?,?,?,?)", [
    ("GOA",       "Goa",         "Beach",     "Goa",              "Nov–Feb"),
    ("GOKARNA",   "Gokarna",     "Beach",     "Karnataka",        "Oct–Mar"),
    ("PONDY",     "Pondicherry", "Beach",     "Puducherry",       "Oct–Mar"),
    ("COORG",     "Coorg",       "Hills",     "Karnataka",        "Oct–Jun"),
    ("OOTY",      "Ooty",        "Hills",     "Tamil Nadu",       "Mar–Jun"),
    ("MANALI",    "Manali",      "Hills",     "Himachal Pradesh", "Mar–Jun"),
    ("HAMPI",     "Hampi",       "Heritage",  "Karnataka",        "Oct–Feb"),
    ("JAIPUR",    "Jaipur",      "Heritage",  "Rajasthan",        "Oct–Mar"),
    ("RISHIKESH", "Rishikesh",   "Adventure", "Uttarakhand",      "Oct–Jun"),
    ("DANDELI",   "Dandeli",     "Adventure", "Karnataka",        "Oct–May"),
])
cur.executemany("INSERT INTO stays VALUES (?,?,?)", [   # one room fits 2 people
    ("GOA",       "Sea Breeze Guesthouse",   3000), ("GOKARNA", "Om Beach Huts",          1800),
    ("PONDY",     "French Quarter Homestay", 2500), ("COORG",   "Coffee Estate Cottage",  2800),
    ("OOTY",      "Nilgiri View Lodge",      2200), ("MANALI",  "Old Manali Cafe Stay",   2500),
    ("HAMPI",     "Boulder View Guesthouse", 1500), ("JAIPUR",  "Pink City Haveli",       2400),
    ("RISHIKESH", "Ganga Riverside Camp",    1600), ("DANDELI", "Jungle River Camp",      2000),
])
cur.executemany("INSERT INTO costs VALUES (?,?,?,?)", [  # travel = return trip per person
    ("GOA",       "Overnight train", 2800, 1500), ("GOKARNA", "Overnight bus",   1200,  900),
    ("PONDY",     "Bus",             1400, 1200), ("COORG",   "Bus",             1000, 1100),
    ("OOTY",      "Bus",             1000, 1000), ("MANALI",  "Flight + bus",    9000, 1300),
    ("HAMPI",     "Overnight train",  900,  800), ("JAIPUR",  "Flight",          7500, 1200),
    ("RISHIKESH", "Flight + cab",    8500, 1400), ("DANDELI", "Bus",             1100, 1600),
])
conn.commit()


def fetch_place(place: str) -> dict | None:
    """JOIN places + stays + costs for one place (by id like 'PONDY' or name like 'Pondicherry')."""
    row = cur.execute("""
        SELECT p.place_id, p.name, p.trip_type, p.state, p.best_months,
               s.stay_name, s.room_price_per_night,
               c.travel_mode, c.travel_cost_per_person, c.daily_spend_per_person
        FROM places p
        JOIN stays s ON s.place_id = p.place_id
        JOIN costs c ON c.place_id = p.place_id
        WHERE UPPER(p.place_id) = UPPER(?) OR UPPER(p.name) = UPPER(?)
    """, (place, place)).fetchone()
    return dict(row) if row else None


def list_places() -> list[dict]:
    """A short list of every destination."""
    return [dict(r) for r in cur.execute(
        "SELECT place_id, name, trip_type, best_months FROM places ORDER BY trip_type, name"
    ).fetchall()]


# =============================================================================
# 2. RAG knowledge base (the travel notebook, 13 tips)
# =============================================================================
KB = [
    ("trip-cost",    "Trip cost per person = travel + stay share + daily spend x nights.",                                  {"topic": "money"}),
    ("rooms",        "One room fits 2 people, so rooms needed = people / 2, rounded up. Stay share = rooms x room price x nights / people.", {"topic": "money"}),
    ("buffer",       "Keep a 10% buffer on top of the total for surprises like a cab surge or a missed bus.",                {"topic": "money"}),
    ("splitting",    "Split shared costs (stay, cabs, fuel) equally. Personal shopping is never split.",                     {"topic": "money"}),
    ("beach",        "Beach trips (Goa, Gokarna, Pondicherry): best November-February. In the monsoon (June-September) many beach shacks and water sports close.", {"topic": "trip_type"}),
    ("hills",        "Hill trips (Coorg, Ooty, Manali): best March-June and October. Avoid July-August: heavy rain and landslides block hill roads. Carry a jacket even in summer.", {"topic": "trip_type"}),
    ("heritage",     "Heritage trips (Hampi, Jaipur): best October-March. April-May afternoons cross 40°C, so explore early in the morning.", {"topic": "trip_type"}),
    ("adventure",    "Adventure trips (Rishikesh, Dandeli): river rafting runs October-June and closes in the monsoon. Use licensed operators and always wear a life jacket.", {"topic": "trip_type"}),
    ("book-early",   "Book trains and flights 30-60 days early; prices often double in the last week.",                     {"topic": "travel_tip"}),
    ("weekdays",     "Weekday stays are often 20-30% cheaper than weekends.",                                                {"topic": "travel_tip"}),
    ("long-weekend", "Long weekends and New Year push Goa prices up sharply; book early or pick Gokarna instead.",           {"topic": "travel_tip"}),
    ("overnight",    "Overnight buses and trains save the cost of one night's stay.",                                       {"topic": "travel_tip"}),
    ("big-group",    "Groups bigger than 6: a villa or homestay is usually cheaper than many hotel rooms.",                 {"topic": "travel_tip"}),
]

embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

chroma = chromadb.Client()
col = chroma.create_collection("trip_tips", metadata={"hnsw:space": "cosine"})
vecs = embed.embed_documents([t for _, t, _ in KB])
col.add(ids=[i for i, _, _ in KB], embeddings=vecs,
        documents=[t for _, t, _ in KB], metadatas=[m for _, _, m in KB])


def retrieve_context(query: str, n: int = 3) -> str:
    """Top-n notebook tips for this question, as bullet points."""
    qv = embed.embed_query(query)
    res = col.query(query_embeddings=[qv], n_results=n)
    return "\n".join(f"- {c}" for c in res["documents"][0])


# =============================================================================
# 3. Tools (the agent decides when to call these)
# =============================================================================
@tool
def db_lookup(place: str) -> dict:
    """Fetch a destination's trip type, best months, stay, room price, travel cost and daily spend from the database."""
    record = fetch_place(place)
    return record if record else {"error": f"{place} not found. Use list_destinations to see valid places."}


@tool
def list_destinations() -> list[dict]:
    """Return all destinations in the database."""
    return list_places()


@tool
def compute_trip_cost(travel_cost_per_person: float, room_price_per_night: float,
                      daily_spend_per_person: float, people: int, nights: int,
                      budget_per_person: float | None = None) -> dict:
    """
    Compute the trip cost per person with exact arithmetic.
    One room fits 2 people. Returns: rooms_needed, stay_share_per_person,
    food_and_fun_per_person, total_per_person, and (if a budget is given)
    within_budget and money_left_per_person (negative = over budget).
    """
    rooms_needed = math.ceil(people / 2)
    stay_share = round(rooms_needed * room_price_per_night * nights / people, 2)
    food_and_fun = round(daily_spend_per_person * nights, 2)
    total = round(travel_cost_per_person + stay_share + food_and_fun, 2)
    result = {
        "rooms_needed": rooms_needed,
        "stay_share_per_person": stay_share,
        "food_and_fun_per_person": food_and_fun,
        "total_per_person": total,
    }
    if budget_per_person is not None:
        result["within_budget"] = total <= budget_per_person
        result["money_left_per_person"] = round(budget_per_person - total, 2)
    return result


TOOLS = [db_lookup, list_destinations, compute_trip_cost]
TOOL_MAP = {t.name: t for t in TOOLS}

# =============================================================================
# 4. Agent
# =============================================================================
MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.ai/v1")
# llm = ChatOpenAI(model=MODEL, reasoning_effort="none", base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY).bind_tools(TOOLS)   # see lesson 04
llm = ChatGroq(model=MODEL,temperature=0,max_tokens=None,timeout=None,max_retries=2)

SYSTEM = """You are Trip Buddy, a cheerful trip planner for groups of friends travelling from Bengaluru.

When the user asks about a trip:
1. Use db_lookup to fetch the place's prices
2. Use compute_trip_cost for exact arithmetic (never do the maths yourself)
3. Give a clear answer: can they afford it, cost per person, and 1-2 useful tips
If the number of friends or nights is missing, ask for it before calculating.

When the user asks to list places, call list_destinations.

Travel notebook (use this for tips and warnings):
{rag_context}"""

MAX_STEPS = 10   # safety net so a confused model can't loop (and bill you) forever


def run_agent(user_message: str) -> str:
    """One full ReAct run: RAG context → (think → tool → observe)* → final answer."""
    # Build RAG context based on the message
    rag_context = retrieve_context(user_message)

    messages = [
        SystemMessage(content=SYSTEM.format(rag_context=rag_context)),
        HumanMessage(content=user_message),
    ]

    # ReAct loop
    for _ in range(MAX_STEPS):
        response = llm.invoke(messages)      # Reason: model decides what to do next
        messages.append(response)

        if not response.tool_calls:          # no tool requested → final answer
            return response.text

        for tc in response.tool_calls:       # Act: run each requested tool
            fn = TOOL_MAP[tc["name"]]
            result = fn.invoke(tc["args"])
            print(f"  [tool] {tc['name']}({tc['args']})")
            messages.append(ToolMessage(     # Observe: hand the result back
                content=json.dumps(result, default=str),
                tool_call_id=tc["id"],
            ))

    return "I had to stop after too many steps. Try asking about one place at a time."


# =============================================================================
# 5. Gradio chat interface
# =============================================================================
def chat_handler(message, _history):
    """Gradio calls this for every chat message. _history is unused (each question stands alone)."""
    try:
        return run_agent(message)
    except Exception as e:
        return (
            f"Something went wrong: {e}\n\n"
            "Try: *\"We are 4 friends with ₹12,000 each. Can we do Goa for 3 nights?\"*  \n"
            "Or: **list** to see all destinations."
        )


demo = gr.ChatInterface(
    fn=chat_handler,
    title="Trip Buddy (DB + RAG + Tools)",
    description=(
        "Plan a trip with friends. The agent fetches prices from the DB, "
        "does exact maths with tools, and uses the travel notebook for tips.  \n"
        "Type **list** to see all destinations."
    ),
    examples=[
        "list",
        "We are 4 friends with ₹12,000 each. Can we do Goa for 3 nights?",
        "Is Manali a good idea in July?",
        "How much is Hampi for 5 friends, 2 nights?",
        "Which trips fit ₹8,000 each for 3 friends, 2 nights?",
    ],
)

if __name__ == "__main__":
    print("Starting Trip Buddy...")
    print(f"  model: {MODEL}")
    print(f"  {len(list_places())} destinations in DB")
    print(f"  {col.count()} tips in ChromaDB")
    demo.launch(debug=True)
