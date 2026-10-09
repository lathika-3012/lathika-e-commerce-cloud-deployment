
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "models/distilbert_model"
VALIDATION_PATH = "data/splits/validation.csv"

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully.")

# --------------------------------------------------
# Load validation data
# --------------------------------------------------

df = pd.read_csv(VALIDATION_PATH)

print(f"Validation samples: {len(df)}")

texts = df["instruction"].tolist()
true_labels = df["label"].tolist()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

predicted_labels = []
confidences = []

print("\nAnalyzing validation data...")

for i, text in enumerate(texts):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence, predicted_id = torch.max(probabilities, dim=-1)

    predicted_labels.append(predicted_id.item())
    confidences.append(confidence.item())

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(texts)}")

# --------------------------------------------------
# Add results to dataframe
# --------------------------------------------------

df["predicted_label"] = predicted_labels
df["confidence"] = confidences

df["correct"] = (
    df["label"] == df["predicted_label"]
)

# --------------------------------------------------
# Overall statistics
# --------------------------------------------------

accuracy = df["correct"].mean()

average_confidence = df["confidence"].mean()

correct_confidence = df.loc[
    df["correct"], "confidence"
].mean()

incorrect_confidence = df.loc[
    ~df["correct"], "confidence"
].mean()

print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

print(f"\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")

print(f"\nAverage Confidence:")
print(f"{average_confidence * 100:.2f}%")

print(f"\nConfidence on Correct Predictions:")
print(f"{correct_confidence * 100:.2f}%")

print(f"\nConfidence on Incorrect Predictions:")
print(f"{incorrect_confidence * 100:.2f}%")

# --------------------------------------------------
# Confidence ranges
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONFIDENCE DISTRIBUTION")
print("=" * 70)

ranges = [
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.01)
]

for lower, upper in ranges:

    count = (
        (df["confidence"] >= lower)
        & (df["confidence"] < upper)
    ).sum()

    percentage = count / len(df) * 100

    print(
        f"{lower * 100:.0f}% - "
        f"{min(upper, 1.0) * 100:.0f}% : "
        f"{count} samples "
        f"({percentage:.2f}%)"
    )

# --------------------------------------------------
# Accuracy at different confidence thresholds
# --------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

for threshold in thresholds:

    selected = df[
        df["confidence"] >= threshold
    ]

    if len(selected) == 0:
        continue

    threshold_accuracy = selected["correct"].mean()

    coverage = len(selected) / len(df)

    print(
        f"\nThreshold >= {threshold:.2f}"
    )

    print(
        f"Samples accepted : {len(selected)}"
    )

    print(
        f"Coverage          : {coverage * 100:.2f}%"
    )

    print(
        f"Accuracy          : {threshold_accuracy * 100:.2f}%"
    )

# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

OUTPUT_PATH = "results/confidence_analysis.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("Analysis completed.")
print(f"Detailed results saved to: {OUTPUT_PATH}")
print("=" * 70)

