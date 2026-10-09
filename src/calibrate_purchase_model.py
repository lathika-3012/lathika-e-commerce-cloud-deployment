import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/purchase_dataset.csv"
MODEL_DIR = "models/purchase_model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "purchase_probability_calibrated.pkl"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATING PURCHASE PROBABILITY MODEL")
print("=" * 70)

import pandas as pd

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded: {len(df)} records")

# ============================================================
# FEATURES
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

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# BASE MODEL
# ============================================================

base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# ============================================================
# CALIBRATION
# ============================================================

calibrated_model = CalibratedClassifierCV(
    estimator=base_pipeline,
    method="sigmoid",
    cv=5
)

print("\nTraining calibrated model...")
calibrated_model.fit(X_train, y_train)

# ============================================================
# EVALUATION
# ============================================================

predictions = calibrated_model.predict(X_test)
probabilities = calibrated_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)
logloss = log_loss(y_test, probabilities)
brier = brier_score_loss(y_test, probabilities)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATED MODEL RESULTS")
print("=" * 70)

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"Log Loss       : {logloss:.4f}")
print(f"Brier Score    : {brier:.4f}")

# ============================================================
# SAMPLE PROBABILITIES
# ============================================================

print("\nSample calibrated probabilities:")

for i in range(min(10, len(probabilities))):
    print(
        f"Customer {i + 1:02d} → "
        f"{probabilities[i] * 100:.2f}%"
    )

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    calibrated_model,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("CALIBRATED MODEL SAVED")
print("=" * 70)

print(f"Model location: {MODEL_PATH}")
print("=" * 70)