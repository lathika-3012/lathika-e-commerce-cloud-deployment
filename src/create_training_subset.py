import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/splits/train.csv"
OUTPUT_FILE = "data/splits/train_subset.csv"

SAMPLES_PER_INTENT = 300
RANDOM_STATE = 42


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Original training samples: {len(df)}")
print(f"Number of intents: {df['intent'].nunique()}")


# ============================================================
# CREATE BALANCED SUBSET
# ============================================================

print("\nCreating balanced training subset...")

# Sample 300 examples from every intent
subset = (
    df.groupby("intent", group_keys=False)
      .sample(
          n=SAMPLES_PER_INTENT,
          random_state=RANDOM_STATE
      )
      .reset_index(drop=True)
)


# ============================================================
# SHUFFLE DATA
# ============================================================

subset = subset.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

subset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VERIFY DATASET
# ============================================================

print("\n========== COMPLETE ==========")

print(f"Original training samples : {len(df)}")
print(f"New training samples      : {len(subset)}")
print(f"Number of intents         : {subset['intent'].nunique()}")

print("\nColumns:")
print(subset.columns.tolist())

print("\nSamples per intent:")
print(subset["intent"].value_counts().sort_index())

print(f"\nSaved to:")
print(OUTPUT_FILE)