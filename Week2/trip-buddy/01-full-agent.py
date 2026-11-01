"""
Trip Buddy Advisor: SQLite + ChromaDB + a tool-calling RAG agent.

Run from this folder with:
    python 01-full-agent.py

Set OPENAI_API_KEY in the environment or in a .env file. The first run also
downloads the sentence-transformer model used to search the local travel notes.
SQLite and ChromaDB data are persisted beside this script.
"""

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
SQLITE_PATH = BASE_DIR / "trip_buddy.db"
CHROMA_PATH = BASE_DIR / "trip_buddy_chroma"
COLLECTION_NAME = "travel_blogs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_STEPS = 8

PLACES = [
    ("GOA", "Goa", "Beach", 2800, 1500, "Sea Breeze Guesthouse", 3000),
    ("COORG", "Coorg", "Hills", 1000, 1100, "Coffee Estate Cottage", 2800),
    ("HAMPI", "Hampi", "Heritage", 3500, 900, "Boulder View Guesthouse", 1500),
    ("MANALI", "Manali", "Hills", 9000, 1300, "Mountain View Homestay", 2500),
]

# These starter travel-blog notes make the project usable immediately. Replace
# or extend them with your own blog content as needed.
TRAVEL_BLOGS = [
    {
        "id": "hill-season",
        "title": "Planning a Hill Trip",
        "category": "season",
        "text": (
            "For hill destinations such as Coorg, Ooty, and Manali, March to June "
            "and October are generally more suitable travel periods. Avoid July "
            "and August where possible because heavy rain can cause landslides "
            "and road closures. Carry rain protection and warm layers."
        ),
    },
    {
        "id": "beach-season",
        "title": "When to Plan a Beach Holiday",
        "category": "season",
        "text": (
            "Goa, Gokarna, and Pondicherry are popular from November to February. "
            "During the monsoon, some beach shacks and water activities may close, "
            "so check local conditions before booking."
        ),
    },
    {
        "id": "heritage-season",
        "title": "Exploring Heritage Destinations",
        "category": "season",
        "text": (
            "Hampi and Jaipur are often most comfortable to explore from October "
            "to March. Start early to see temples, forts, and outdoor sites before "
            "the afternoon heat, especially in April and May."
        ),
    },
    {
        "id": "group-rooms",
        "title": "Splitting Accommodation Costs",
        "category": "budget",
        "text": (
            "For a group of friends, estimate one room for every two people and "
            "round the number of rooms up. Divide the total room cost by the "
            "number of travelers to estimate each person's share."
        ),
    },
    {
        "id": "booking-budget",
        "title": "Keeping a Trip on Budget",
        "category": "budget",
        "text": (
            "Estimate each person's total as travel cost plus their share of the "
            "stay plus daily spending multiplied by the number of nights. Keep "
            "extra room in the budget for local transport and unexpected costs. "
            "Booking trains and flights 30 to 60 days ahead can help control fares."
        ),
    },
    {
        "id": "adventure-season",
        "title": "Planning an Adventure Trip",
        "category": "season",
        "text": (
            "River rafting around Rishikesh and Dandeli generally operates from "
            "October through June and may close during the monsoon. Confirm "
            "operator schedules and safety conditions before traveling."
        ),
    },
]

embeddings: HuggingFaceEmbeddings | None = None
blog_collection: Any = None


def initialize_sqlite() -> None:
    """Create the destination tables and add starter data without overwriting it."""
    with sqlite3.connect(SQLITE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS places (
                place_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                trip_type TEXT NOT NULL,
                travel_cost_per_person REAL NOT NULL,
                daily_spend_per_person REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stays (
                place_id TEXT PRIMARY KEY,
                stay_name TEXT NOT NULL,
                room_price_per_night REAL NOT NULL,
                FOREIGN KEY (place_id) REFERENCES places(place_id)
            );
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO places
                (place_id, name, trip_type, travel_cost_per_person, daily_spend_per_person)
            VALUES (?, ?, ?, ?, ?)
            """,
            [place[:5] for place in PLACES],
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO stays (place_id, stay_name, room_price_per_night)
            VALUES (?, ?, ?)
            """,
            [(place[0], place[5], place[6]) for place in PLACES],
        )


def initialize_chromadb() -> None:
    """Persist the starter blog notes in ChromaDB and prepare semantic search."""
    global embeddings, blog_collection

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    blog_collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(blog_collection.get()["ids"])
    new_blogs = [blog for blog in TRAVEL_BLOGS if blog["id"] not in existing_ids]

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if new_blogs:
        blog_collection.add(
            ids=[blog["id"] for blog in new_blogs],
            embeddings=embeddings.embed_documents(
                [blog["text"] for blog in new_blogs]
            ),
            documents=[blog["text"] for blog in new_blogs],
            metadatas=[
                {"title": blog["title"], "category": blog["category"]}
                for blog in new_blogs
            ],
        )


def get_place(place: str) -> dict[str, Any] | None:
    """Read a destination and its stay from SQLite."""
    with sqlite3.connect(SQLITE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT p.place_id, p.name, p.trip_type, p.travel_cost_per_person,
                   p.daily_spend_per_person, s.stay_name, s.room_price_per_night
            FROM places AS p
            LEFT JOIN stays AS s ON s.place_id = p.place_id
            WHERE UPPER(p.place_id) = UPPER(?) OR UPPER(p.name) = UPPER(?)
            """,
            (place.strip(), place.strip()),
        ).fetchone()
    return dict(row) if row else None


@tool
def lookup_place(place: str) -> dict[str, Any]:
    """Look up a destination's trip type, travel and daily costs, and stay details."""
    result = get_place(place)
    if result is None:
        with sqlite3.connect(SQLITE_PATH) as connection:
            places = connection.execute(
                "SELECT name FROM places ORDER BY name"
            ).fetchall()
        return {
            "error": f"{place!r} was not found.",
            "available_places": [row[0] for row in places],
        }
    return result


@tool
def search_travel_blogs(query: str, n: int = 3) -> list[dict[str, Any]]:
    """Search travel-blog notes for relevant destination, season, or budget advice."""
    if blog_collection is None or embeddings is None:
        raise RuntimeError("Travel blog search is not initialized.")
    if not 1 <= n <= 6:
        return [{"error": "n must be between 1 and 6."}]

    results = blog_collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=min(n, blog_collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        {
            "title": metadata["title"],
            "category": metadata["category"],
            "content": document,
            "distance": round(distance, 4),
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


@tool
def estimate_trip_cost(place: str, people: int, nights: int) -> dict[str, Any]:
    """Estimate per-person trip cost from SQLite destination prices and group size."""
    if people <= 0 or nights <= 0:
        return {"error": "people and nights must both be greater than zero."}

    destination = get_place(place)
    if destination is None:
        return lookup_place.invoke({"place": place})
    if destination["room_price_per_night"] is None:
        return {"error": f"No stay price is recorded for {destination['name']}."}

    rooms_needed = math.ceil(people / 2)
    stay_share = (
        rooms_needed * destination["room_price_per_night"] * nights / people
    )
    daily_spend = destination["daily_spend_per_person"] * nights
    total = destination["travel_cost_per_person"] + stay_share + daily_spend
    return {
        "place": destination["name"],
        "people": people,
        "nights": nights,
        "rooms_needed": rooms_needed,
        "stay_share_per_person": round(stay_share, 2),
        "travel_cost_per_person": destination["travel_cost_per_person"],
        "daily_spend_per_person": round(daily_spend, 2),
        "estimated_total_per_person": round(total, 2),
    }


TOOLS = [lookup_place, search_travel_blogs, estimate_trip_cost]
TOOL_MAP = {registered_tool.name: registered_tool for registered_tool in TOOLS}


def build_agent() -> Any:
    """Create the LLM with the SQLite and ChromaDB tools attached."""
    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            f"Set GROQ_API_KEY in the environment or in {BASE_DIR / '.env'}."
        )
    model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
    return ChatOpenAI(model=model, reasoning_effort="none").bind_tools(TOOLS)


def run_agent(agent: Any, user_question: str) -> str:
    """Run the tool-calling RAG loop until the advisor returns its answer."""
    messages = [
        HumanMessage(
            content=(
                "You are Trip Buddy Advisor. Help the user plan a practical trip. "
                "For destination-specific advice, look up the place in SQLite and "
                "search the travel blogs in ChromaDB. Use the cost-estimate tool "
                "when the question includes a group size and number of nights. "
                "Clearly identify estimates and mention when requested information "
                "is not available; do not invent database facts.\n\n"
                f"User request: {user_question}"
            )
        )
    ]

    for _ in range(MAX_STEPS):
        response = agent.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.text

        for tool_call in response.tool_calls:
            selected_tool = TOOL_MAP.get(tool_call["name"])
            if selected_tool is None:
                raise ValueError(f"Unknown tool requested: {tool_call['name']}")
            result = selected_tool.invoke(tool_call["args"])
            print(f"[tool] {tool_call['name']} -> {result}")
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )

    raise RuntimeError(f"Agent exceeded the limit of {MAX_STEPS} tool-call steps.")


def main() -> None:
    """Initialize local stores and answer one trip-planning question."""
    initialize_sqlite()
    initialize_chromadb()
    agent = build_agent()

    print("Trip Buddy Advisor (type 'quit' to exit)")
    while True:
        question = input("\nHow can I help plan your trip? ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        print("\n" + run_agent(agent, question))


if __name__ == "__main__":
    main()