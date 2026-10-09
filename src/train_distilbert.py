import os
import json
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "data/splits/train_subset.csv"
VALIDATION_FILE = "data/splits/validation.csv"
TEST_FILE = "data/splits/test.csv"

MODEL_OUTPUT = "models/distilbert_model"
RESULTS_FILE = "results/distilbert_results.json"

# Fast CPU training settings
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(validation_df)}")
print(f"Test samples       : {len(test_df)}")


# ============================================================
# 3. LOAD LABEL MAPPING
# ============================================================

print("\nLoading label mapping...")

with open("data/processed/label_mapping.json", "r") as file:
    label_mapping = json.load(file)


# Your JSON can be in either format:
#
# {"0": "add_product", "1": "availability", ...}
#
# OR
#
# {"add_product": 0, "availability": 1, ...}


# Handle nested label_mapping.json
if "label2id" in label_mapping and "id2label" in label_mapping:

    label2id = {
        str(label): int(label_id)
        for label, label_id in label_mapping["label2id"].items()
    }

    id2label = {
        int(label_id): str(label)
        for label_id, label in label_mapping["id2label"].items()
    }

else:
    raise ValueError(
        "label_mapping.json does not contain "
        "'label2id' and 'id2label'."
    )

NUM_LABELS = len(id2label)

print(f"Number of intents : {NUM_LABELS}")

print("\nIntent labels:")

for label_id in sorted(id2label):
    print(f"{label_id}: {id2label[label_id]}")

NUM_LABELS = len(id2label)

print(f"Number of intents : {NUM_LABELS}")

print("\nIntent labels:")

for label_id in sorted(id2label):
    print(f"{label_id}: {id2label[label_id]}")


# ============================================================
# 4. CONVERT DATA TO HUGGING FACE DATASETS
# ============================================================

print("\nConverting datasets...")

train_dataset = Dataset.from_pandas(
    train_df[["instruction", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["instruction", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["instruction", "label"]],
    preserve_index=False
)


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING DISTILBERT TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 6. TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["instruction"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("\nTokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenizing validation data...")

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenizing test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenization completed.")


# ============================================================
# 7. LOAD DISTILBERT MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING DISTILBERT MODEL")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

print("DistilBERT loaded successfully.")


# ============================================================
# 8. METRICS
# ============================================================

def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    }


# ============================================================
# 9. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 10. TRAINING ARGUMENTS
# ============================================================

print("\nSetting training configuration...")

training_args = TrainingArguments(

    output_dir=MODEL_OUTPUT,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    logging_steps=100,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=1,

    report_to="none",

    fp16=False,

    dataloader_num_workers=0,

    seed=42
)


# ============================================================
# 11. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# 12. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING DISTILBERT TRAINING")
print("=" * 60)

print(f"Model           : {MODEL_NAME}")
print(f"Training samples: {len(train_df)}")
print(f"Number of labels: {NUM_LABELS}")
print(f"Epochs          : {EPOCHS}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Max length      : {MAX_LENGTH}")
print(f"Learning rate   : {LEARNING_RATE}")

print("\nPlease wait...\n")

training_result = trainer.train()


# ============================================================
# 13. VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=validation_dataset
)

for key, value in validation_results.items():
    print(f"{key}: {value}")


# ============================================================
# 14. TEST
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)

for key, value in test_results.items():
    print(f"{key}: {value}")


# ============================================================
# 15. SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs(MODEL_OUTPUT, exist_ok=True)

trainer.save_model(MODEL_OUTPUT)

tokenizer.save_pretrained(MODEL_OUTPUT)

print(f"Model saved to: {MODEL_OUTPUT}")


# ============================================================
# 16. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results = {

    "model": MODEL_NAME,

    "number_of_labels": NUM_LABELS,

    "train_samples": len(train_df),

    "validation_samples": len(validation_df),

    "test_samples": len(test_df),

    "epochs": EPOCHS,

    "batch_size": BATCH_SIZE,

    "max_length": MAX_LENGTH,

    "learning_rate": LEARNING_RATE,

    "validation_results": validation_results,

    "test_results": test_results
}


with open(RESULTS_FILE, "w") as file:

    json.dump(
        results,
        file,
        indent=4,
        default=float
    )


# ============================================================
# 17. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("DISTILBERT TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(MODEL_OUTPUT)

print("\nResults:")
print(RESULTS_FILE)

print("\nYou can now use the trained DistilBERT model.")