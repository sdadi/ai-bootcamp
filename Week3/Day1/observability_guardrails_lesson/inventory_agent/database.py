"""
database.py  -  the inventory database  (by Shivank Agarwal)

Creates a small SQLite database (a single file, inventory.db) with one table:

    product_name | current_stock | average_demand (units/day) | lead_time (days)

Versions v2 to v7 call init_database() at start-up, so the file is created
automatically the first time you run them.
"""
import sqlite3

DB_PATH = "inventory.db"

SAMPLE_DATA = [
    # (product, current stock, average daily demand, lead time in days)
    ("toothpaste", 20, 5.0, 4),
    ("shampoo", 15, 3.0, 5),
    ("soap", 50, 8.0, 3),
    ("detergent", 25, 4.5, 6),
    ("toilet_paper", 100, 12.0, 7),
    ("toothbrush", 30, 2.0, 3),
    ("mouthwash", 10, 1.0, 4),
    ("floss", 20, 1.5, 5),
    ("razor", 15, 0.5, 3),
    ("shaving_cream", 25, 1.0, 4),
    ("shaving_gel", 10, 0.5, 3),
    ("shaving_foam", 15, 1.0, 4),
    ("milk", 10, 2.0, 3),
    ("bread", 20, 1.0, 4),
    ("eggs", 30, 0.5, 3),
    ("cheese", 15, 1.0, 4),
    ("butter", 10, 0.5, 3),
    ("yogurt", 15, 1.0, 4),
    ("juice", 20, 2.0, 3),
]


def init_database(db_path: str = DB_PATH) -> bool:
    """Create the table (if missing) and add the sample products (if missing)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY,
                product_name TEXT UNIQUE NOT NULL,
                current_stock INTEGER NOT NULL,
                average_demand REAL NOT NULL,
                lead_time INTEGER NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.executemany("""
            INSERT OR IGNORE INTO inventory (product_name, current_stock, average_demand, lead_time)
            VALUES (?, ?, ?, ?)
        """, SAMPLE_DATA)
        conn.commit()      # save the changes to the file
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        return False


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection to the database."""
    return sqlite3.connect(db_path)
