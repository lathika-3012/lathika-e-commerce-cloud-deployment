
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/purchase_dataset.csv"
MODEL_PATH = "models/purchase_model"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PURCHASE PROBABILITY MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total samples: {len(df)}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

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

TARGET = "purchase"


X = df[FEATURES]
y = df[TARGET]


print("\nFeatures:")
for feature in FEATURES:
    print(f"- {feature}")

print(f"\nTarget: {TARGET}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nData split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# MODEL
# ============================================================

# StandardScaler normalizes the numerical features.
#
# Logistic Regression then learns:
#
# P(purchase = 1 | customer behavior)
#
# predict_proba() will later give us the
# purchase probability needed by the router.

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining purchase probability model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Probability of purchase = class 1
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

loss = log_loss(
    y_test,
    y_probability
)


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Log Loss : {loss:.4f}")


# ============================================================
# SHOW SAMPLE PURCHASE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PURCHASE PROBABILITIES")
print("=" * 70)

sample_data = X_test.head(10)

sample_probabilities = model.predict_proba(
    sample_data
)[:, 1]

for i, probability in enumerate(
    sample_probabilities,
    start=1
):

    print(
        f"Customer {i}: "
        f"{probability:.4f} "
        f"({probability * 100:.2f}%)"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

model_file = os.path.join(
    MODEL_PATH,
    "purchase_probability_model.pkl"
)

joblib.dump(
    model,
    model_file
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"\nModel path:")
print(model_file)

print("\nPurchase probability model training completed.")

