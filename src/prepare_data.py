import os
import json
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
)

data = dataset["train"]

print("Dataset loaded!")
print("Original number of records:", len(data))


# ============================================================
# 2. CONVERT TO PANDAS
# ============================================================

df = data.to_pandas()

# We only need these two columns for intent classification
df = df[["instruction", "intent"]]


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

print(df.isnull().sum())

df = df.dropna(subset=["instruction", "intent"])


# ============================================================
# 4. CLEAN TEXT
# ============================================================

df["instruction"] = df["instruction"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()


# Remove empty queries
df = df[df["instruction"] != ""]


# ============================================================
# 5. REMOVE DUPLICATE QUERIES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["instruction", "intent"]
)

after_duplicates = len(df)

print("\nDuplicate removal:")
print("Before:", before_duplicates)
print("After :", after_duplicates)
print("Removed:", before_duplicates - after_duplicates)


# ============================================================
# 6. DISPLAY INTENT INFORMATION
# ============================================================

print("\n========== INTENT INFORMATION ==========")

print("Number of intents:", df["intent"].nunique())

print("\nIntent distribution:")

print(
    df["intent"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. CREATE LABEL MAPPING
# ============================================================

labels = sorted(df["intent"].unique())

label2id = {
    label: index
    for index, label in enumerate(labels)
}

id2label = {
    index: label
    for index, label in enumerate(labels)
}


print("\n========== LABEL MAPPING ==========")

for label, index in label2id.items():
    print(index, "->", label)


# ============================================================
# 8. ADD NUMERIC LABEL
# ============================================================

df["label"] = df["intent"].map(label2id)


# ============================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating train/validation/test split...")

# First split:
# 80% training
# 20% temporary

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


# Second split:
# 10% validation
# 10% test

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# ============================================================
# 10. PRINT SPLIT INFORMATION
# ============================================================

print("\n========== DATASET SPLIT ==========")

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))

print("\nExpected ratio:")
print("Training   = 80%")
print("Validation = 10%")
print("Testing    = 10%")


# ============================================================
# 11. CREATE DIRECTORIES
# ============================================================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)

# ============================================================
# 12. SAVE PROCESSED DATA
# ============================================================

df.to_csv(
    "data/processed/intent_dataset.csv",
    index=False
)


# ============================================================
# 13. SAVE TRAIN / VALIDATION / TEST
# ============================================================

train_df.to_csv(
    "data/splits/train.csv",
    index=False
)

validation_df.to_csv(
    "data/splits/validation.csv",
    index=False
)

test_df.to_csv(
    "data/splits/test.csv",
    index=False
)


# ============================================================
# 14. SAVE LABEL MAPPING
# ============================================================

with open(
    "data/processed/label_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label2id": label2id,
            "id2label": id2label
        },
        f,
        indent=4
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n========== COMPLETE ==========")

print("Processed dataset:")
print("data/processed/intent_dataset.csv")

print("\nTraining dataset:")
print("data/splits/train.csv")

print("\nValidation dataset:")
print("data/splits/validation.csv")

print("\nTest dataset:")
print("data/splits/test.csv")

print("\nLabel mapping:")
print("data/processed/label_mapping.json")

print("\nDataset preparation completed successfully!")