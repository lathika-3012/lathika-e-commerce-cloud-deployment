from datasets import load_dataset

print("Loading dataset...")

ds = load_dataset(
    "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
)

print("\n========== DATASET ==========")
print(ds)

print("\n========== SPLITS ==========")
print(ds.keys())

for split in ds:
    print(f"\n========== {split.upper()} ==========")

    print("Number of rows:", len(ds[split]))

    print("Columns:")
    print(ds[split].column_names)

    print("\nFirst 3 records:")

    for i in range(min(3, len(ds[split]))):
        print(f"\nRecord {i + 1}:")
        print(ds[split][i])