
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert_model"

print("Loading DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully.")


def predict_top5(text):
    # Convert customer query into tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits into probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    # Get top 5 predictions
    top5_probabilities, top5_ids = torch.topk(probabilities, 5)

    results = []

    for probability, class_id in zip(top5_probabilities, top5_ids):

        class_id = class_id.item()
        probability = probability.item()

        intent = model.config.id2label[class_id]

        results.append({
            "intent": intent,
            "confidence": probability
        })

    return results


# Test queries
test_queries = [
    "Where is my order?",
    "I want to return the product I bought",
    "Can I cancel my order?",
    "My payment failed",
    "I forgot my password",
    "What payment methods do you accept?",
    "The product I received is damaged",
    "I want to speak with a human",
]


print("\n" + "=" * 70)
print("DISTILBERT TOP-5 INTENT PREDICTIONS")
print("=" * 70)


for query in test_queries:

    results = predict_top5(query)

    print("\nCustomer Query:")
    print(query)

    print("\nTop 5 Predictions:")

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. "
            f"{result['intent']:<35} "
            f"{result['confidence'] * 100:.2f}%"
        )

    print("-" * 70)


# Interactive mode
print("\nInteractive mode started.")
print("Type 'exit' to stop.\n")


while True:

    query = input("Customer: ")

    if query.lower() == "exit":
        print("Prediction stopped.")
        break

    results = predict_top5(query)

    print("\nTop 5 Predictions:")

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. "
            f"{result['intent']:<35} "
            f"{result['confidence'] * 100:.2f}%"
        )

    print()

