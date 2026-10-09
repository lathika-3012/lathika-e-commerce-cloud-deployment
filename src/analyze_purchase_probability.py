
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)
from sklearn.calibration import calibration_curve


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/purchase_dataset.csv"
MODEL_PATH = "models/purchase_model/purchase_probability_model.pkl"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PURCHASE PROBABILITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "product_views",
    "cart_additions",
    "previous_purchases",
    "previous_sessions",
    "favorites",
    "time_spent_minutes",
    "cart_value",
    "checkout_attempts",
    "customer_tier"
]

X = df[FEATURES]
y = df["purchase"]


# Use the same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading purchase model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# BASIC METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

logloss = log_loss(
    y_test,
    probabilities
)

brier = brier_score_loss(
    y_test,
    probabilities
)


print("\n" + "=" * 70)
print("PROBABILITY QUALITY")
print("=" * 70)

print(f"\nROC-AUC  : {roc_auc:.4f}")
print(f"Log Loss : {logloss:.4f}")
print(f"Brier    : {brier:.4f}")


# ============================================================
# CALIBRATION BINS
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_test,
    probabilities,
    n_bins=10,
    strategy="uniform"
)


print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS")
print("=" * 70)

print(
    "\nPredicted Probability | Actual Purchase Rate"
)

print("-" * 55)

for predicted, actual in zip(
    prob_pred,
    prob_true
):

    print(
        f"{predicted * 100:>8.2f}%"
        f"                 "
        f"{actual * 100:>8.2f}%"
    )


# ============================================================
# PURCHASE PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PURCHASE PROBABILITY DISTRIBUTION")
print("=" * 70)

ranges = [
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.01)
]

for lower, upper in ranges:

    mask = (
        (probabilities >= lower)
        & (probabilities < upper)
    )

    count = mask.sum()

    if count == 0:
        print(
            f"{lower:.1f} - {min(upper, 1.0):.1f}"
            f" : 0 customers"
        )
        continue

    actual_purchase_rate = y_test.iloc[
        mask
    ].mean()

    print(
        f"{lower:.1f} - {min(upper, 1.0):.1f}"
        f" : {count} customers"
        f" | Actual purchase rate:"
        f" {actual_purchase_rate * 100:.2f}%"
    )


# ============================================================
# HOT / WARM / COLD PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("HOT / WARM / COLD PREVIEW")
print("=" * 70)

# Proposed thresholds for inspection only.
# We are NOT finalizing these yet.

hot = probabilities >= 0.80

warm = (
    (probabilities >= 0.40)
    & (probabilities < 0.80)
)

cold = probabilities < 0.40


print(
    f"\nHOT  (>= 0.80): "
    f"{hot.sum()} customers "
    f"({hot.mean() * 100:.2f}%)"
)

print(
    f"WARM (0.40-0.80): "
    f"{warm.sum()} customers "
    f"({warm.mean() * 100:.2f}%)"
)

print(
    f"COLD (< 0.40): "
    f"{cold.sum()} customers "
    f"({cold.mean() * 100:.2f}%)"
)


# ============================================================
# EXAMPLE CUSTOMERS
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE PURCHASE PROBABILITIES")
print("=" * 70)

for i, probability in enumerate(
    probabilities[:10],
    start=1
):

    if probability >= 0.80:
        category = "HOT"

    elif probability >= 0.40:
        category = "WARM"

    else:
        category = "COLD"

    print(
        f"Customer {i}: "
        f"{probability * 100:.2f}%"
        f" -> {category}"
    )


print("\n" + "=" * 70)
print("PURCHASE PROBABILITY ANALYSIS COMPLETED")
print("=" * 70)

