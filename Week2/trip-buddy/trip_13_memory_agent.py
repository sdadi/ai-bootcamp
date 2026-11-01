"""
trip_13_memory_agent.py
───────────────────────
Lesson 13 · Trip Buddy that REMEMBERS you: lesson 08 + short-term and long-term memory.

The story
    Lesson 08's Trip Buddy has no memory at all. Ask "And for 5 friends?" and it has
    no idea what you were talking about. Close the app and it forgets your group forever.
    This version fixes both:
      SHORT-TERM   the chat history is sent with every question, so follow-ups work
                   ("Can 4 of us do Goa for 3 nights?" ... "And for 5 friends?")
      LONG-TERM    click "End chat & save to memory": the pipeline from lesson 12
                   extracts facts, past trips and rules, and files them on disk
      RECALL       every new question starts by loading what Trip Buddy remembers
                   into the system prompt, so it never asks for your group size twice

What is new compared with lesson 08
    - chat_handler uses `history` (lesson 08 ignored it)
    - the system prompt has a "what you remember" block (RECALL)
    - two buttons: "End chat & save to memory" (the pipeline) and "Forget everything"
    - a panel showing the filing cabinet: semantic / episodic / procedural
    Database, travel notebook (RAG), tools and ReAct loop are the same as lesson 08.

How to run
    uv run trip_13_memory_agent.py        (needs OPENAI_API_KEY in .env)
    then open http://127.0.0.1:7860

Try this demo (it shows every kind of memory)
    1. "Hi! We're 4 friends from Bengaluru, budget ₹12,000 each. Aman is vegetarian."
    2. "Last March we loved Gokarna, but the overnight bus made Neha sick."
    3. "Always add a 10% buffer for us, we overspend."
    4. "Can we do Goa for 3 nights?"          ← no need to repeat group size or budget
    5. "And for 5 friends?"                    ← short-term memory makes the follow-up work
    6. Click "End chat & save to memory"       ← watch the memory panel fill up
    7. Stop the app (Ctrl+C), start it again, and ask "Plan a 2-night beach trip for us."
       It still knows your group: that's long-term memory.
"""

import inspect
import json
import math
import os
import sqlite3

import chromadb
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from trip_memory import LongTermMemory, save_session, text_of, trim

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
# 4. Memory (new in this lesson)
# =============================================================================
ltm = LongTermMemory(embed=embed)        # the filing cabinet on disk: memory_store/ (shared with lesson 12)
SHORT_TERM_LIMIT = 10                    # how many recent chat messages we send with each question

# =============================================================================
# 5. Agent
# =============================================================================
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
base_llm = ChatOpenAI(model=MODEL, reasoning_effort="none")   # plain model, used by the memory pipeline
llm = base_llm.bind_tools(TOOLS)                              # model + tools, used by the agent (see lesson 04)

SYSTEM = """You are Trip Buddy, a cheerful trip planner for groups of friends travelling from Bengaluru.

When the user asks about a trip:
1. Use db_lookup to fetch the place's prices
2. Use compute_trip_cost for exact arithmetic (never do the maths yourself)
3. Give a clear answer: can they afford it, cost per person, and 1-2 useful tips
Use what you remember about this group: do not ask for details you already know
(like group size or budget), and follow their planning rules.
If the number of friends or nights is still unknown, ask for it before calculating.
When the user asks to list places, call list_destinations.

What you remember about this group (long-term memory):
{memory}

Travel notebook (use this for tips and warnings):
{rag_context}"""

MAX_STEPS = 10   # safety net so a confused model can't loop (and bill you) forever


def run_agent(user_message: str, history: list[dict]) -> str:
    """ReAct run with memory: RECALL (long-term) + chat history (short-term) + the new message."""
    messages = [SystemMessage(content=SYSTEM.format(memory=ltm.recall(user_message),          # long-term
                                                    rag_context=retrieve_context(user_message)))]
    for m in trim(history, SHORT_TERM_LIMIT):                                                  # short-term
        text = text_of(m["content"])
        messages.append(HumanMessage(content=text) if m["role"] == "user" else AIMessage(content=text))
    messages.append(HumanMessage(content=user_message))

    for _ in range(MAX_STEPS):                                   # ReAct loop (same as lesson 08)
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.text
        for tc in response.tool_calls:
            result = TOOL_MAP[tc["name"]].invoke(tc["args"])
            print(f"  [tool] {tc['name']}({tc['args']})")
            messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc["id"]))

    return "I had to stop after too many steps. Try asking about one place at a time."


# =============================================================================
# 6. Gradio app: chat on the left, the filing cabinet on the right
# =============================================================================
def respond(message: str, history: list):
    """Send a message. `history` is the short-term memory Gradio keeps for us."""
    history = history or []
    if not message.strip():
        return "", history
    try:
        reply = run_agent(message, history)
    except Exception as e:
        reply = f"Something went wrong: {e}"
    return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]


def end_chat_and_save(history: list):
    """THE PIPELINE: short-term (this chat) → long-term (the filing cabinet). Then clear the chat."""
    chat = [{"role": m["role"], "content": text_of(m["content"])} for m in (history or [])]
    if not chat:
        return "Nothing to save: the chat is empty.", ltm.as_markdown(), []
    items, report = save_session(chat, ltm, base_llm)
    lines = [f"**Saved from {len(chat)} chat messages:**"]
    lines += [f"- `{r['action']}` {r['type']}: {r['text']}" + (f" (was: {r['old']})" if r.get("old") else "")
              for r in report] or ["- nothing worth remembering"]
    print("[memory] pipeline:", [(r["action"], r["type"]) for r in report])
    return "\n".join(lines), ltm.as_markdown(), []          # [] = new chat, short-term memory is empty


def forget_everything():
    ltm.forget_everything()
    return "All long-term memory deleted.", ltm.as_markdown()


# Gradio 5 needs type="messages"; Gradio 6 always uses this format, so the option was removed.
CHATBOT_OPTIONS = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}

with gr.Blocks(title="Trip Buddy with Memory") as demo:
    gr.Markdown("## Trip Buddy with Memory (DB + RAG + Tools + Memory)\n"
                "Chat normally. Click **End chat & save to memory** to move what matters from this chat "
                "(short-term) into the filing cabinet (long-term). Restart the app: Trip Buddy still remembers.")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=460, label="Short-term memory: this chat", **CHATBOT_OPTIONS)
            box = gr.Textbox(placeholder="e.g. Hi! We're 4 friends from Bengaluru, budget ₹12,000 each.",
                             show_label=False)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                save_btn = gr.Button("End chat & save to memory")
        with gr.Column(scale=2):
            gr.Markdown("### Long-term memory (saved on disk)")
            memory_panel = gr.Markdown(ltm.as_markdown())
            pipeline_report = gr.Markdown()
            forget_btn = gr.Button("Forget everything", variant="stop")

    box.submit(respond, [box, chatbot], [box, chatbot])
    send_btn.click(respond, [box, chatbot], [box, chatbot])
    save_btn.click(end_chat_and_save, [chatbot], [pipeline_report, memory_panel, chatbot])
    forget_btn.click(forget_everything, None, [pipeline_report, memory_panel])

if __name__ == "__main__":
    print("Starting Trip Buddy with Memory...")
    print(f"  model: {MODEL}")
    print(f"  {len(list_places())} destinations in DB, {col.count()} tips in ChromaDB")
    print(f"  long-term memory: {len(ltm.all_facts())} facts, {len(ltm.all_episodes())} past trips, "
          f"{len(ltm.all_rules())} rules  (folder: memory_store/)")
    demo.launch(debug=True)
