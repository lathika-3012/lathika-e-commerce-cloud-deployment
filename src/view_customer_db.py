import sqlite3
from pathlib import Path


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "customer.db"


# ============================================================
# CONNECT TO DATABASE
# ============================================================

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()


# ============================================================
# DISPLAY DATABASE INFORMATION
# ============================================================

print("=" * 120)
print("                    CUSTOMER DATABASE VIEWER")
print("=" * 120)

# Count customers
cursor.execute("SELECT COUNT(*) FROM customers")
total_customers = cursor.fetchone()[0]

print(f"\nDatabase : {DB_PATH}")
print(f"Total customers : {total_customers}")


# ============================================================
# DISPLAY ALL CUSTOMERS
# ============================================================

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
ORDER BY customer_id
""")

rows = cursor.fetchall()


print("\n" + "=" * 120)
print("ALL CUSTOMER RECORDS")
print("=" * 120)

# Column headings
print(
    f"{'Customer ID':<12}"
    f"{'Views':<8}"
    f"{'Cart':<8}"
    f"{'Purchases':<12}"
    f"{'Sessions':<10}"
    f"{'Favorites':<10}"
    f"{'Time(min)':<12}"
    f"{'Cart Value':<13}"
    f"{'Checkout':<10}"
    f"{'Tier':<6}"
)

print("-" * 120)


# Display rows
for row in rows:

    print(
        f"{row[0]:<12}"
        f"{row[1]:<8}"
        f"{row[2]:<8}"
        f"{row[3]:<12}"
        f"{row[4]:<10}"
        f"{row[5]:<10}"
        f"{row[6]:<12.1f}"
        f"₹{row[7]:<12.1f}"
        f"{row[8]:<10}"
        f"{row[9]:<6}"
    )


# ============================================================
# SEARCH FOR A CUSTOMER
# ============================================================

print("\n" + "=" * 120)
print("SEARCH CUSTOMER")
print("=" * 120)

customer_id = input(
    "\nEnter Customer ID to view details "
    "(example: CUST0001, or press Enter to skip): "
).strip().upper()


if customer_id:

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
    WHERE customer_id = ?
    """, (customer_id,))

    customer = cursor.fetchone()

    if customer:

        print("\n" + "=" * 60)
        print(f"PROFILE: {customer[0]}")
        print("=" * 60)

        print(f"Product Views       : {customer[1]}")
        print(f"Cart Additions      : {customer[2]}")
        print(f"Previous Purchases  : {customer[3]}")
        print(f"Previous Sessions   : {customer[4]}")
        print(f"Favorites           : {customer[5]}")
        print(f"Time Spent          : {customer[6]} minutes")
        print(f"Cart Value          : ₹{customer[7]}")
        print(f"Checkout Attempts   : {customer[8]}")

        tier_names = {
            0: "New",
            1: "Regular",
            2: "Premium"
        }

        print(
            f"Customer Tier      : "
            f"{tier_names.get(customer[9], 'Unknown')}"
        )

    else:

        print(f"\nCustomer '{customer_id}' was not found.")


# ============================================================
# CLOSE DATABASE
# ============================================================

connection.close()

print("\n" + "=" * 120)
print("Database viewer finished.")
print("=" * 120)