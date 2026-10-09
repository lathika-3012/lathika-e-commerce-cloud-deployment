import sqlite3
from pathlib import Path


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    BASE_DIR
    / "data"
    / "customer.db"
)


# ============================================================
# GET CUSTOMER PROFILE
# ============================================================

def get_customer_profile(customer_id):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
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
            customer_tier,
            current_order_value
        FROM customers
        WHERE customer_id = ?
        """,
        (customer_id,)
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    customer = get_customer_profile(
        "CUST0999"
    )

    if customer is None:

        print(
            "Customer not found."
        )

    else:

        print(
            "\nCustomer profile retrieved successfully:\n"
        )

        for key, value in customer.items():

            print(
                f"{key:<25} : {value}"
            )