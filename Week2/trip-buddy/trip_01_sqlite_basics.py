"""
trip_01_sqlite_basics.py
────────────────────────
Lesson 01 · SQLite basics: save our trip options in a database.

The story
    Every friend group has "the Goa plan" that never happens. This time we
    plan properly. First we need one place that remembers each destination
    and where we would stay. That place is a database.

What you learn
    - Create two tables (places, stays) linked by place_id (a foreign key)
    - Insert many rows at once with executemany
    - JOIN the two tables to see everything about a place in one row
    - row_factory = sqlite3.Row, so you can write row["name"] instead of row[1]

How to run
    uv run trip_01_sqlite_basics.py
    (no API key needed, SQLite comes built into Python)

What you will see
    A small table of 3 places, then a lookup for "hampi" and for a missing "PARIS".
"""

import sqlite3

# ":memory:" = the database lives in RAM and disappears when the script ends.
# Perfect for learning: every run starts fresh.
conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row   # lets us do row["column_name"]
cur = conn.cursor()              # the cursor is the "pen" that reads/writes the DB

# ── 1. Create the tables ──────────────────────────────────────────────────────
# places = where we could go
# stays  = where we would sleep there, and the price of one room per night
cur.executescript("""
CREATE TABLE places (
    place_id  TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    trip_type TEXT NOT NULL
);

CREATE TABLE stays (
    place_id             TEXT PRIMARY KEY,
    stay_name            TEXT NOT NULL,
    room_price_per_night INTEGER NOT NULL,
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);
""")

# ── 2. Add 3 places (executemany = one call, many rows) ───────────────────────
cur.executemany("INSERT INTO places VALUES (?, ?, ?)", [
    ("GOA",   "Goa",   "Beach"),
    ("COORG", "Coorg", "Hills"),
    ("HAMPI", "Hampi", "Heritage"),
])

cur.executemany("INSERT INTO stays VALUES (?, ?, ?)", [
    ("GOA",   "Sea Breeze Guesthouse",  3000),
    ("COORG", "Coffee Estate Cottage",  2800),
    ("HAMPI", "Boulder View Guesthouse", 1500),
])

conn.commit()

# ── 3. JOIN: combine both tables into one view ────────────────────────────────
print(f"{'ID':<8} {'Place':<8} {'Type':<10} {'Stay':<26} {'Room/night':>10}")
print("-" * 66)

rows = cur.execute("""
    SELECT p.place_id, p.name, p.trip_type,
           s.stay_name, s.room_price_per_night
    FROM   places p
    JOIN   stays s ON s.place_id = p.place_id
    ORDER  BY p.place_id
""").fetchall()

for r in rows:
    print(f"{r['place_id']:<8} {r['name']:<8} {r['trip_type']:<10} {r['stay_name']:<26} {r['room_price_per_night']:>10}")


# ── 4. A reusable lookup function ─────────────────────────────────────────────
def fetch_place(place: str) -> dict | None:
    """Return one place (with its stay) as a dict, or None if not found.

    UPPER(...) on both sides means "hampi" and "HAMPI" both work.
    The "?" placeholder keeps the query safe from SQL injection.
    """
    row = cur.execute(
        "SELECT p.*, s.stay_name, s.room_price_per_night "
        "FROM places p JOIN stays s ON s.place_id = p.place_id "
        "WHERE UPPER(p.place_id) = UPPER(?)",
        (place,)
    ).fetchone()
    return dict(row) if row else None


print("\nLookup 'hampi':", fetch_place("hampi"))
print("Lookup 'PARIS':", fetch_place("PARIS"))
