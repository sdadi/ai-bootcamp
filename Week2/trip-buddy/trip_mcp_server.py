"""
trip_mcp_server.py
──────────────────
The Trip Buddy MCP server. It holds the database and offers it to any AI app through MCP.

The story
    In lesson 08 the tools lived inside the agent's own file. Now we move them into
    a separate MCP server. Any MCP-compatible app (our lesson 10 agent, Claude Desktop,
    Cursor, VS Code...) can now plug in and use Trip Buddy's data without custom code.

What it offers (the 3 MCP building blocks)
    Tools      the LLM decides when to call them      @mcp.tool()
        db_lookup, list_destinations, compute_trip_cost   (same 3 tools as lesson 08)
    Resources  read-only data at a URI; the APP decides when to read them   @mcp.resource()
        trips://destinations          all places with trip type and best months
        trips://place/{place_id}      one place's full record (a URI template)
    Prompts    a ready-made message the USER picks    @mcp.prompt()
        plan_trip(place, people, nights, budget_per_person)

How to run
    You normally do NOT run this file yourself. Lessons 09 and 10 start it for you
    as a subprocess and talk to it over stdin/stdout ("stdio" transport).
    If you run it directly (uv run trip_mcp_server.py) it just waits silently for an
    MCP client. Press Ctrl+C to stop.

Important rule for stdio servers
    Never print() here. stdout is the wire that carries the MCP messages, so any
    extra text would break the conversation with the client.
"""

import json
import math
import sqlite3
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("Trip Buddy Server", log_level="WARNING")   # WARNING = hide the routine "Processing request" logs

# =============================================================================
# 1. Database (same 3 tables and 10 places as lesson 08)
#    Only the server touches the database now. The agent never sees SQL.
# =============================================================================
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
# 2. TOOLS: the LLM decides when to call these
#    FastMCP turns the type hints into a JSON Schema and the docstring into the
#    description. That description is ALL the LLM knows about the tool, so be clear.
# =============================================================================
@mcp.tool()
def db_lookup(
    place: Annotated[str, Field(description="Place name like 'Goa' or id like 'PONDY'")],
) -> dict:
    """Fetch a destination's trip type, best months, stay, room price, travel cost and daily spend from the database."""
    record = fetch_place(place)
    return record if record else {"error": f"{place} not found. Use list_destinations to see valid places."}


@mcp.tool()
def list_destinations() -> list[dict]:
    """Return all destinations in the database."""
    return list_places()


@mcp.tool()
def compute_trip_cost(
    travel_cost_per_person: Annotated[float, Field(ge=0, description="Return travel cost for one person, in rupees")],
    room_price_per_night: Annotated[float, Field(ge=0, description="Price of one room for one night (a room fits 2)")],
    daily_spend_per_person: Annotated[float, Field(ge=0, description="Food, local travel and fun per person per day")],
    people: Annotated[int, Field(ge=1, le=20, description="Number of friends travelling (1-20)")],
    nights: Annotated[int, Field(ge=1, le=30, description="Number of nights (1-30)")],
    budget_per_person: Annotated[float | None, Field(description="Optional budget per person, in rupees")] = None,
) -> dict:
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


# =============================================================================
# 3. RESOURCES: read-only data at a URI. The APP decides when to read them,
#    not the LLM. Our agent (lesson 10) reads trips://destinations at the start
#    of every chat and puts it in the system prompt.
# =============================================================================
@mcp.resource("trips://destinations", mime_type="application/json")
def destinations_resource() -> str:
    """All destinations with their trip type and best months."""
    return json.dumps(list_places(), ensure_ascii=False)


@mcp.resource("trips://place/{place_id}", mime_type="application/json")
def place_resource(place_id: str) -> str:
    """One destination's full record, e.g. trips://place/HAMPI (a URI template)."""
    return json.dumps(fetch_place(place_id) or {"error": f"{place_id} not found"}, ensure_ascii=False)


# =============================================================================
# 4. PROMPTS: a ready-made message the USER picks (like a slash command).
#    The LLM never calls a prompt by itself. MCP sends prompt arguments as text.
# =============================================================================
@mcp.prompt()
def plan_trip(place: str, people: str, nights: str, budget_per_person: str) -> str:
    """Plan a trip: fills the group's details into a well-phrased question."""
    return (
        f"We are {people} friends with ₹{budget_per_person} each. "
        f"Can we do {place} for {nights} nights? "
        "Give the cost per person, say whether it fits our budget, and add 2 useful tips."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")   # stdio = the client starts us and talks over stdin/stdout
