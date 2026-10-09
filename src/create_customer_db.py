import sqlite3
import random
from pathlib import Path


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "customer.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE CONNECTION
# ============================================================

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()


# ============================================================
# CREATE CUSTOMER TABLE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    product_views INTEGER NOT NULL,
    cart_additions INTEGER NOT NULL,
    previous_purchases INTEGER NOT NULL,
    previous_sessions INTEGER NOT NULL,
    favorites INTEGER NOT NULL,
    time_spent_minutes REAL NOT NULL,
    cart_value REAL NOT NULL,
    checkout_attempts INTEGER NOT NULL,
    customer_tier INTEGER NOT NULL
)
""")


# ============================================================
# GENERATE CUSTOMER DATA
# ============================================================

NUM_CUSTOMERS = 1000

customers = []

random.seed(42)

for i in range(1, NUM_CUSTOMERS + 1):

    customer_id = f"CUST{i:04d}"

    # --------------------------------------------------------
    # Customer behavior
    # --------------------------------------------------------

    customer_type = random.choice([
        "low",
        "medium",
        "high"
    ])

    if customer_type == "low":

        product_views = random.randint(1, 8)
        cart_additions = random.randint(0, 2)
        previous_purchases = random.randint(0, 1)
        previous_sessions = random.randint(1, 4)
        favorites = random.randint(0, 2)
        time_spent = random.randint(2, 10)
        cart_value = random.randint(0, 1000)
        checkout_attempts = random.randint(0, 1)
        customer_tier = 0

    elif customer_type == "medium":

        product_views = random.randint(8, 20)
        cart_additions = random.randint(2, 5)
        previous_purchases = random.randint(1, 4)
        previous_sessions = random.randint(4, 10)
        favorites = random.randint(1, 4)
        time_spent = random.randint(10, 30)
        cart_value = random.randint(500, 4000)
        checkout_attempts = random.randint(1, 2)
        customer_tier = random.choice([0, 1])

    else:

        product_views = random.randint(15, 40)
        cart_additions = random.randint(4, 10)
        previous_purchases = random.randint(3, 10)
        previous_sessions = random.randint(8, 20)
        favorites = random.randint(3, 8)
        time_spent = random.randint(20, 60)
        cart_value = random.randint(3000, 10000)
        checkout_attempts = random.randint(1, 4)
        customer_tier = random.choice([1, 2])

    customers.append((
        customer_id,
        product_views,
        cart_additions,
        previous_purchases,
        previous_sessions,
        favorites,
        time_spent,
        cart_value,
        checkout_attempts,
        customer_tier
    ))


# ============================================================
# INSERT DATA
# ============================================================

cursor.executemany("""
INSERT OR REPLACE INTO customers (
    customer_id,
    product_views,
    cart_additions,
    previous_purchases,
    previous_sessions,
    favorites,
    time_spent_minutes,
    cart_value,
    checkout_attempts,
    customer_tier
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", customers)


# ============================================================
# SAVE DATABASE
# ============================================================

connection.commit()


# ============================================================
# VERIFY DATABASE
# ============================================================

cursor.execute("SELECT COUNT(*) FROM customers")
total_customers = cursor.fetchone()[0]

print("=" * 60)
print("CUSTOMER DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Database location : {DB_PATH}")
print(f"Customers stored  : {total_customers}")

print("\nSample customer records:")
print("-" * 60)

cursor.execute("""
SELECT
    customer_id,
    product_views,
    cart_additions,
    previous_purchases,
    previous_sessions,
    favorites,
    time_spent_minutes,
    cart_value,
    checkout_attempts,
    customer_tier
FROM customers
LIMIT 5
""")

rows = cursor.fetchall()

for row in rows:
    print(row)

print("=" * 60)


# ============================================================
# CLOSE DATABASE
# ============================================================

connection.close()