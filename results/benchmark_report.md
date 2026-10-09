# Benchmark Report

_Generated 2026-10-08T05:40:56+00:00 - mode: **quick** - device: **CPU**_

All values below were measured by `benchmark_6_models.py` on this machine. Latency depends on the hardware and device, so compare models only within the same report.

## 1. Dataset
- Dataset: `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset`
- Text column: `instruction`  |  Label column: `intent`
- Samples after cleaning: 44884 (used in this run: 1500)
- Number of classes: 46

## 2. Train/test split
- 1200 training / 300 test samples (test_size=0.2, random_state=42, stratified where possible)
- The split is made before any TF-IDF fitting or tokenisation; all models use the identical split.

## 3. Models and metrics
Six approaches were attempted; the table lists those that finished.

| Model | Accuracy | Weighted F1 | Precision | Recall | Latency (ms/query) | Size (MB) | Train time (s) | Parameters | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9567 | 0.9553 | 0.9624 | 0.9567 | 0.71 | 1.17 | 0.4 | 143,612 | 0.998 |
| Linear SVM | 0.9500 | 0.9488 | 0.9547 | 0.9500 | 0.34 | 1.17 | 0.0 | 143,612 | 0.995 |
| DistilBERT | 0.2667 | 0.2300 | 0.3952 | 0.2667 | 38.02 | 255.55 | 94.0 | 66,988,846 | 0.216 |
| MobileBERT | 0.0533 | 0.0365 | 0.0621 | 0.0533 | 37.26 | 93.99 | 81.8 | 24,605,486 | 0.099 |
| RoBERTa Base | 0.5600 | 0.4935 | 0.5422 | 0.5600 | 41.02 | 475.64 | 182.1 | 124,681,006 | 0.367 |

**Models that failed in this run:**
- BERT Tiny: ValueError: Couldn't instantiate the backend tokenizer from one of: 
(1) a `tokenizers` library serialization file, 
(2) a slow tokenizer instance to convert or 
(3) an equivalent slow tokenizer class to instantiate and convert. 
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one..

## 4. Selection methodology
`score = 0.40*norm(F1) + 0.30*norm(Accuracy) + 0.20*norm_inverse(Latency) + 0.10*norm_inverse(Size)`

Each term is min-max normalised across the models in this run (latency and size are inverted so smaller is better). Weights: weighted_f1=0.4, accuracy=0.3, latency_inverse=0.2, size_inverse=0.1.

## 5. Best model
- Overall highest selection score: **Logistic Regression**
- Best by weighted F1: **Logistic Regression**

## 6. Latency comparison (fastest first)
- Linear SVM: 0.34 ms/query
- Logistic Regression: 0.71 ms/query
- MobileBERT: 37.26 ms/query
- DistilBERT: 38.02 ms/query
- RoBERTa Base: 41.02 ms/query

## 7. Model size comparison (smallest first)
- Linear SVM: 1.17 MB
- Logistic Regression: 1.17 MB
- MobileBERT: 93.99 MB
- DistilBERT: 255.55 MB
- RoBERTa Base: 475.64 MB

## 8. Final production model
- **RoBERTa Base**
- RoBERTa Base has the best selection score among transformers (0.3672); the preferred model 'DistilBERT' was not available or scored too far below it.

The production model is restricted to transformers (a compact SLM); the overall winner is reported separately and honestly above.
