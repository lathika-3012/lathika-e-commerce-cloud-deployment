# ============================================================
# CREATE SYNTHETIC PURCHASE DATASET
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
NUM_CUSTOMERS = 10000

np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "purchase_dataset.csv"
)


# ============================================================
# GENERATE CUSTOMER BEHAVIOR
# ============================================================

customers = []

for _ in range(NUM_CUSTOMERS):

    # --------------------------------------------------------
    # Customer activity level
    #
    # We intentionally create a mixture of:
    # LOW, MEDIUM and HIGH activity customers.
    # --------------------------------------------------------

    activity = np.random.choice(
        ["low", "medium", "high"],
        p=[0.35, 0.40, 0.25]
    )

    # --------------------------------------------------------
    # LOW ACTIVITY
    # --------------------------------------------------------

    if activity == "low":

        product_views = np.random.randint(
            1, 12
        )

        cart_additions = np.random.randint(
            0, 3
        )

        previous_purchases = np.random.randint(
            0, 3
        )

        previous_sessions = np.random.randint(
            1, 7
        )

        favorites = np.random.randint(
            0, 4
        )

        time_spent_minutes = np.random.uniform(
            2,
            20
        )

        cart_value = np.random.uniform(
            0,
            2500
        )

        checkout_attempts = np.random.randint(
            0,
            2
        )

        customer_tier = np.random.choice(
            [0, 1],
            p=[0.75, 0.25]
        )

    # --------------------------------------------------------
    # MEDIUM ACTIVITY
    # --------------------------------------------------------

    elif activity == "medium":

        product_views = np.random.randint(
            8, 30
        )

        cart_additions = np.random.randint(
            1, 7
        )

        previous_purchases = np.random.randint(
            1, 6
        )

        previous_sessions = np.random.randint(
            4, 15
        )

        favorites = np.random.randint(
            1, 9
        )

        time_spent_minutes = np.random.uniform(
            15,
            60
        )

        cart_value = np.random.uniform(
            500,
            6000
        )

        checkout_attempts = np.random.randint(
            0,
            3
        )

        customer_tier = np.random.choice(
            [0, 1, 2],
            p=[0.15, 0.65, 0.20]
        )

    # --------------------------------------------------------
    # HIGH ACTIVITY
    # --------------------------------------------------------

    else:

        product_views = np.random.randint(
            20, 60
        )

        cart_additions = np.random.randint(
            4, 12
        )

        previous_purchases = np.random.randint(
            3, 12
        )

        previous_sessions = np.random.randint(
            10, 30
        )

        favorites = np.random.randint(
            5, 15
        )

        time_spent_minutes = np.random.uniform(
            40,
            120
        )

        cart_value = np.random.uniform(
            2000,
            10000
        )

        checkout_attempts = np.random.randint(
            1,
            5
        )

        customer_tier = np.random.choice(
            [1, 2],
            p=[0.60, 0.40]
        )


    # ========================================================
    # PURCHASE PROPENSITY
    # ========================================================
    #
    # Instead of directly making most customers purchase,
    # we calculate a behavioral score and convert it into
    # a probability.
    #
    # This gives us both purchasers and non-purchasers.
    # ========================================================

    score = (

        -2.0

        + 0.035 * product_views

        + 0.22 * cart_additions

        + 0.10 * previous_purchases

        + 0.025 * previous_sessions

        + 0.06 * favorites

        + 0.018 * time_spent_minutes

        + 0.00008 * cart_value

        + 0.45 * checkout_attempts

        + 0.30 * customer_tier
    )


    # --------------------------------------------------------
    # Convert score to probability
    # --------------------------------------------------------

    purchase_probability = (
        1 /
        (
            1 +
            np.exp(-score)
        )
    )


    # --------------------------------------------------------
    # Add a small amount of randomness.
    #
    # This prevents the dataset from becoming perfectly
    # separable.
    # --------------------------------------------------------

    purchase_probability = np.clip(
        purchase_probability,
        0.05,
        0.95
    )


    # --------------------------------------------------------
    # Generate purchase outcome
    # --------------------------------------------------------

    purchase = np.random.binomial(
        1,
        purchase_probability
    )


    customers.append({

        "product_views":
            product_views,

        "cart_additions":
            cart_additions,

        "previous_purchases":
            previous_purchases,

        "previous_sessions":
            previous_sessions,

        "favorites":
            favorites,

        "time_spent_minutes":
            round(
                time_spent_minutes,
                2
            ),

        "cart_value":
            round(
                cart_value,
                2
            ),

        "checkout_attempts":
            checkout_attempts,

        "customer_tier":
            customer_tier,

        "purchase":
            purchase
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    customers
)


# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SYNTHETIC PURCHASE DATASET CREATED"
)

print(
    "=" * 70
)

print(
    f"Dataset location : {OUTPUT_PATH}"
)

print(
    f"Total customers  : {len(df)}"
)


purchase_counts = (
    df["purchase"]
    .value_counts()
    .sort_index()
)


print(
    "\nPurchase Distribution:"
)

print(
    f"No Purchase (0) : "
    f"{purchase_counts.get(0, 0)}"
)

print(
    f"Purchase (1)    : "
    f"{purchase_counts.get(1, 0)}"
)


purchase_rate = (
    df["purchase"].mean()
    * 100
)


print(
    f"\nPurchase Rate    : "
    f"{purchase_rate:.2f}%"
)


print(
    "\nFirst 5 rows:"
)

print(
    df.head()
)


print(
    "\n"
    + "=" * 70
)

print(
    "DATASET CREATION COMPLETED"
)

print(
    "=" * 70
)