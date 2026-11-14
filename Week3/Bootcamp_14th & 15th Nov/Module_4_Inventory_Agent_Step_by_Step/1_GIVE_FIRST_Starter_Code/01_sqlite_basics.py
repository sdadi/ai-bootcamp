"""
Step 01 — SQLite basics: store and query inventory data
Created by Shivank Agarwal

Goal: keep structured data (products, suppliers) in a real SQL database and read it back.
SQLite comes built into Python, so there is nothing to install.

Run it:
    python 01_sqlite_basics.py
"""

import sqlite3

# ":memory:" = a temporary database that lives in RAM and disappears when the script ends
conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row  # lets us read columns by name: row["sku_name"]
cur = conn.cursor()             # the "pen" we use to run SQL

# 1) Create two tables linked by sku_id (a FOREIGN KEY)
cur.executescript("""
CREATE TABLE products (
    sku_id   TEXT PRIMARY KEY,
    sku_name TEXT NOT NULL,
    stock    INTEGER NOT NULL
);

CREATE TABLE suppliers (
    sku_id         TEXT PRIMARY KEY,
    supplier_name  TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL,   -- days between placing an order and receiving it
    FOREIGN KEY (sku_id) REFERENCES products(sku_id)
);
""")

# 2) Insert rows. executemany runs the same INSERT once per tuple. "?" = safe placeholder.
cur.executemany("INSERT INTO products VALUES (?, ?, ?)", [
    ("SKU-001", "Wireless Headphones", 150),
    ("SKU-002", "Paracetamol 500mg",   800),
    ("SKU-003", "Instant Noodles",     300),
])
cur.executemany("INSERT INTO suppliers VALUES (?, ?, ?)", [
    ("SKU-001", "TechImports Ltd", 21),
    ("SKU-002", "MedSupply Co",     7),
    ("SKU-003", "FoodDist Pvt",     4),
])
conn.commit()

# 3) Read both tables together with a JOIN
# ----- TODO 1: Write a JOIN query that returns sku_id, sku_name, stock, lead_time_days, supplier_name for every product -----
# Hint: SELECT p.sku_id, p.sku_name, p.stock, s.lead_time_days, s.supplier_name
# Hint: FROM products p JOIN suppliers s ON s.sku_id = p.sku_id ORDER BY p.sku_id
# Hint: cur.execute(sql).fetchall() returns all rows
rows = []  # replace with your code

print(f"{'SKU':<10} {'Name':<22} {'Stock':>6} {'Lead (days)':>12}  Supplier")
print("-" * 70)
for r in rows:
    print(f"{r['sku_id']:<10} {r['sku_name']:<22} {r['stock']:>6} {r['lead_time_days']:>12}  {r['supplier_name']}")


# 4) A reusable lookup function: one SKU in, one dict out (or None)
def fetch_product(sku_id: str) -> dict | None:
    # ----- TODO 2: Fetch ONE product (joined with its supplier) by sku_id; return it as a dict, or None if not found -----
    # Hint: Same JOIN as above + WHERE UPPER(p.sku_id) = UPPER(?)   (so "sku-002" also works)
    # Hint: cur.execute(sql, (sku_id,)).fetchone() returns one row or None
    # Hint: dict(row) turns a Row into a normal dict
    return None  # replace with your code


print("\nLookup sku-002:", fetch_product("sku-002"))
print("Lookup SKU-999:", fetch_product("SKU-999"))
