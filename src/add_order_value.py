import sqlite3
import random
from pathlib import Path


# ============================================================
# DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "customer.db"


connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()


# ============================================================
# ADD CURRENT ORDER VALUE COLUMN
# ============================================================

cursor.execute("PRAGMA table_info(customers)")
columns = [row[1] for row in cursor.fetchall()]

if "current_order_value" not in columns:

    cursor.execute("""
        ALTER TABLE customers
        ADD COLUMN current_order_value REAL DEFAULT 0
    """)

    print("current_order_value column added.")

else:

    print("current_order_value already exists.")


# ============================================================
# GENERATE ORDER VALUE
# ============================================================

random.seed(42)

cursor.execute("""
    SELECT customer_id, cart_value
    FROM customers
""")

customers = cursor.fetchall()

for customer_id, cart_value in customers:

    if cart_value > 0:

        current_order_value = round(
            random.uniform(
                max(100, cart_value * 0.30),
                cart_value
            ),
            2
        )

    else:

        current_order_value = 0.0

    cursor.execute("""
        UPDATE customers
        SET current_order_value = ?
        WHERE customer_id = ?
    """, (
        current_order_value,
        customer_id
    ))


connection.commit()


# ============================================================
# VERIFY
# ============================================================

cursor.execute("""
    SELECT
        customer_id,
        cart_value,
        current_order_value
    FROM customers
    LIMIT 10
""")

rows = cursor.fetchall()

print("\n" + "=" * 75)
print("ORDER VALUE ADDED")
print("=" * 75)

print(
    f"{'Customer ID':<15}"
    f"{'Cart Value':<20}"
    f"{'Current Order Value':<25}"
)

print("-" * 75)

for row in rows:

    print(
        f"{row[0]:<15}"
        f"₹{row[1]:<19.2f}"
        f"₹{row[2]:<24.2f}"
    )

print("=" * 75)

connection.close()