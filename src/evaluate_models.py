import json
import os

# ============================================================
# RESULT FILES
# ============================================================

MODEL_RESULTS = {
    "RoBERTa": "results/roberta_results.json",
    "ELECTRA-Small": "results/electra_results.json",
    "MobileBERT": "results/mobilebert_results.json"
}


# ============================================================
# LOAD RESULTS
# ============================================================

results = []

for model_name, file_path in MODEL_RESULTS.items():

    if os.path.exists(file_path):

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results.append({
            "Model": model_name,
            "Accuracy": data.get("accuracy"),
            "Precision": data.get("precision"),
            "Recall": data.get("recall"),
            "Macro F1": data.get("f1")
        })

    else:

        results.append({
            "Model": model_name,
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "Macro F1": None
        })


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================================")
print("              MODEL COMPARISON")
print("==============================================================")

print(
    f"{'Model':<18}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'Macro F1':<12}"
)

print("-" * 66)


for result in results:

    def format_value(value):

        if value is None:
            return "Pending"

        return f"{value:.4f}"

    print(
        f"{result['Model']:<18}"
        f"{format_value(result['Accuracy']):<12}"
        f"{format_value(result['Precision']):<12}"
        f"{format_value(result['Recall']):<12}"
        f"{format_value(result['Macro F1']):<12}"
    )


print("\n==============================================================")