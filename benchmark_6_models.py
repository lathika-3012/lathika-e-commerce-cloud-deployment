"""Benchmark six NLP intent classifiers, select the best, and export the production model.

Usage (Windows PowerShell)::

    python benchmark_6_models.py --quick     # small subset, 1 epoch  (default, safe for beginners)
    python benchmark_6_models.py --full      # full dataset, TRAINING_EPOCHS epochs

Every number written to ``results/`` is measured by this script - nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import (BASE_DIR, BENCHMARK_CSV_PATH, BENCHMARK_JSON_PATH, BENCHMARK_MODELS_DIR,
                    BENCHMARK_REPORT_PATH, CLASSIFICATION_REPORT_DIR, CONFUSION_MATRIX_DIR, Settings,
                    configure_logging, detect_device, ensure_directories, get_settings)
from dataset_utils import DataBundle, DatasetLoadError, prepare_data, save_label_maps

logger = logging.getLogger("benchmark")


@dataclass(frozen=True)
class ModelSpec:
    """Description of one benchmarked model."""

    key: str
    name: str
    kind: str  # "classical" or "transformer"
    hf_name: Optional[str] = None


def build_model_specs(settings: Settings) -> List[ModelSpec]:
    """The six required models (DistilBERT's checkpoint comes from FALLBACK_MODEL)."""
    return [
        ModelSpec("logistic_regression", "Logistic Regression", "classical"),
        ModelSpec("linear_svm", "Linear SVM", "classical"),
        ModelSpec("distilbert", "DistilBERT", "transformer", settings.FALLBACK_MODEL),
        ModelSpec("bert_tiny", "BERT Tiny", "transformer", "prajjwal1/bert-tiny"),
        ModelSpec("mobilebert", "MobileBERT", "transformer", "google/mobilebert-uncased"),
        ModelSpec("roberta_base", "RoBERTa Base", "transformer", "roberta-base"),
    ]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (and therefore scikit-learn's default RNG users)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy plus weighted F1 / precision / recall."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def measure_latency_ms(predict_one: Callable[[str], Any], texts: Sequence[str], warmup: int,
                       sync: Optional[Callable[[], None]] = None) -> float:
    """Average single-query latency in ms: warm-up first, then one timed call per text."""
    sync = sync or (lambda: None)
    for text in list(texts)[: max(warmup, 0)]:
        predict_one(text)
    sync()
    timings = []
    for text in texts:
        start = time.perf_counter()
        predict_one(text)
        sync()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings)) if timings else float("nan")


def size_mb(paths: Sequence[Path]) -> float:
    """Total size of the given files in MiB."""
    return sum(p.stat().st_size for p in paths if p.exists()) / (1024 * 1024)


def save_diagnostics(spec: ModelSpec, y_true: Sequence[int], y_pred: Sequence[int],
                     id2label: Dict[int, str]) -> None:
    """Save the classification report and confusion matrix for one model."""
    ids = sorted(id2label)
    names = [id2label[i] for i in ids]
    CLASSIFICATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=ids, target_names=names, zero_division=0)
    (CLASSIFICATION_REPORT_DIR / f"{spec.key}.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=ids)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(CONFUSION_MATRIX_DIR / f"{spec.key}.csv")


def _result_row(spec: ModelSpec, metrics: Dict[str, float], latency_ms: float, model_size: float,
                train_time: float, parameters: int, device: str, mode: str, epochs: int,
                bundle: DataBundle, latency_samples: int) -> Dict[str, Any]:
    return {"model": spec.name, "key": spec.key, "type": spec.kind, "hf_name": spec.hf_name or "",
            **metrics, "latency_ms": latency_ms, "size_mb": model_size, "train_time_s": train_time,
            "parameters": int(parameters), "device": device, "mode": mode, "epochs": epochs,
            "n_train": len(bundle.train_df), "n_test": len(bundle.test_df), "latency_samples": latency_samples}


# --------------------------------------------------------------------------- #
# Classical models (TF-IDF + linear classifier)
# --------------------------------------------------------------------------- #
def train_classical(spec: ModelSpec, bundle: DataBundle, settings: Settings, mode: str) -> Dict[str, Any]:
    """Train + evaluate Logistic Regression or Linear SVM on the shared split.

    TF-IDF is fitted inside the pipeline on the training texts only (no leakage).
    """
    seed = settings.RANDOM_SEED
    classifier = (LogisticRegression(C=10.0, max_iter=1000, random_state=seed)
                  if spec.key == "logistic_regression" else LinearSVC(C=1.0, max_iter=5000, random_state=seed))
    pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                         ("clf", classifier)])
    x_train, x_test = bundle.train_df["text"].tolist(), bundle.test_df["text"].tolist()
    y_train = bundle.train_df["label"].map(bundle.label2id).to_numpy()
    y_test = bundle.test_df["label"].map(bundle.label2id).to_numpy()

    print("Training...")
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    print("Evaluating...")
    y_pred = pipeline.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)
    save_diagnostics(spec, y_test, y_pred, bundle.id2label)

    BENCHMARK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = BENCHMARK_MODELS_DIR / f"{spec.key}.joblib"
    joblib.dump(pipeline, model_file)

    samples = x_test[: settings.LATENCY_SAMPLES]
    latency = measure_latency_ms(lambda t: pipeline.predict([t]), samples, settings.LATENCY_WARMUP)
    clf = pipeline.named_steps["clf"]
    parameters = int(clf.coef_.size + clf.intercept_.size)  # TF-IDF vocabulary not counted
    return _result_row(spec, metrics, latency, size_mb([model_file]), train_time, parameters, "cpu", mode,
                       0, bundle, len(samples))


# --------------------------------------------------------------------------- #
# Transformer models (plain PyTorch fine-tuning loop - no deprecated Trainer API)
# --------------------------------------------------------------------------- #
def train_transformer(spec: ModelSpec, bundle: DataBundle, settings: Settings, device: str, mode: str,
                      resume: bool = False) -> Dict[str, Any]:
    """Fine-tune and evaluate one Hugging Face sequence-classification model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    seed = settings.RANDOM_SEED
    epochs = settings.QUICK_EPOCHS if mode == "quick" else settings.TRAINING_EPOCHS
    learning_rate = settings.BERT_TINY_LEARNING_RATE if spec.key == "bert_tiny" else settings.LEARNING_RATE
    num_labels = len(bundle.label2id)

    print(f"Downloading/loading tokenizer '{spec.hf_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    print(f"Downloading/loading model '{spec.hf_name}' "
          "(a 'classifier weights newly initialised' notice is normal)...")
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_name, num_labels=num_labels, id2label=bundle.id2label, label2id=bundle.label2id)
    model.to(device)
    parameters = int(sum(p.numel() for p in model.parameters()))

    # ---- tokenise train / test ------------------------------------------- #
    print("Tokenizing train and test data...")
    train_texts, test_texts = bundle.train_df["text"].tolist(), bundle.test_df["text"].tolist()
    y_train = bundle.train_df["label"].map(bundle.label2id).tolist()
    y_test = bundle.test_df["label"].map(bundle.label2id).to_numpy()
    encoded = tokenizer(train_texts, truncation=True, max_length=settings.MAX_LENGTH)
    items = [{**{k: encoded[k][i] for k in encoded.keys()}, "labels": y_train[i]} for i in range(len(train_texts))]

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        features = [{k: v for k, v in b.items() if k != "labels"} for b in batch]
        padded = tokenizer.pad(features, return_tensors="pt")
        padded["labels"] = labels
        return padded

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(items, batch_size=settings.BATCH_SIZE, shuffle=True,
                                         collate_fn=collate, generator=generator)
    total_steps = max(epochs * len(loader), 1)
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": settings.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(settings.WARMUP_RATIO * total_steps), total_steps)

    # ---- optional resume from the last finished epoch ---------------------- #
    checkpoint_file = BENCHMARK_MODELS_DIR / "checkpoints" / mode / spec.key / "checkpoint.pt"
    start_epoch, elapsed_before = 0, 0.0
    if resume and checkpoint_file.exists():
        state = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, elapsed_before = int(state["epoch"]), float(state["elapsed"])
        print(f"Resuming from checkpoint after epoch {start_epoch}/{epochs}.")

    # ---- train --------------------------------------------------------------- #
    print(f"Training for {epochs} epoch(s), batch size {settings.BATCH_SIZE}, lr {learning_rate}, "
          f"{len(loader)} steps/epoch on {device.upper()}...")
    train_start = time.perf_counter()
    log_every = max(len(loader) // 5, 1)
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
            if step % log_every == 0 or step == len(loader):
                print(f"  epoch {epoch + 1}/{epochs}  step {step}/{len(loader)}  avg loss {running / step:.4f}")
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                    "elapsed": elapsed_before + time.perf_counter() - train_start}, checkpoint_file)
    train_time = elapsed_before + time.perf_counter() - train_start

    # ---- evaluate ------------------------------------------------------------ #
    print("Evaluating...")
    model.eval()
    predictions: List[int] = []
    with torch.no_grad():
        for i in range(0, len(test_texts), settings.EVAL_BATCH_SIZE):
            enc = tokenizer(test_texts[i:i + settings.EVAL_BATCH_SIZE], padding=True, truncation=True,
                            max_length=settings.MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            predictions.extend(model(**enc).logits.argmax(dim=-1).cpu().tolist())
    y_pred = np.array(predictions)
    metrics = compute_metrics(y_test, y_pred)
    save_diagnostics(spec, y_test, y_pred, bundle.id2label)

    # ---- latency (batch size 1, warm-up first) -------------------------------- #
    def predict_one(text: str) -> int:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=settings.MAX_LENGTH)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return int(model(**enc).logits.argmax(dim=-1).item())

    sync = torch.cuda.synchronize if device == "cuda" else None
    samples = test_texts[: settings.LATENCY_SAMPLES]
    latency = measure_latency_ms(predict_one, samples, settings.LATENCY_WARMUP, sync)

    # ---- save model + size ---------------------------------------------------- #
    model_dir = BENCHMARK_MODELS_DIR / spec.key
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    weight_files = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
    model_size = size_mb(weight_files) or size_mb([p for p in model_dir.iterdir() if p.is_file()])

    if checkpoint_file.exists():  # training finished -> checkpoint no longer needed
        checkpoint_file.unlink()
    del model, optimizer, scheduler
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return _result_row(spec, metrics, latency, model_size, train_time, parameters, device, mode, epochs,
                       bundle, len(samples))


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def _minmax(values: Sequence[float], higher_is_better: bool) -> List[float]:
    """Scale to 0..1 (1 = best). If all values are equal every model gets 1.0."""
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    return [((v - low) if higher_is_better else (high - v)) / (high - low) for v in values]


def compute_selection_scores(results: List[Dict[str, Any]], settings: Settings) -> None:
    """Add ``selection_score`` to each result (in place).

    selection_score = 0.40*norm(F1) + 0.30*norm(accuracy)
                    + 0.20*norm_inverse(latency) + 0.10*norm_inverse(size)

    Each term is min-max normalised across the benchmarked models; latency and size
    are inverted so smaller is better. Weights come from the settings.
    """
    weights = (settings.WEIGHT_F1, settings.WEIGHT_ACCURACY, settings.WEIGHT_LATENCY, settings.WEIGHT_SIZE)
    total = sum(weights) or 1.0
    f1_n = _minmax([r["weighted_f1"] for r in results], True)
    acc_n = _minmax([r["accuracy"] for r in results], True)
    lat_n = _minmax([r["latency_ms"] for r in results], False)
    size_n = _minmax([r["size_mb"] for r in results], False)
    for i, row in enumerate(results):
        score = (weights[0] * f1_n[i] + weights[1] * acc_n[i] + weights[2] * lat_n[i] + weights[3] * size_n[i]) / total
        row["selection_score"] = round(float(score), 4)


def choose_models(results: List[Dict[str, Any]], settings: Settings) -> Dict[str, Any]:
    """Pick the overall winner and the production transformer.

    * overall winner       = highest selection score among all models (reported honestly)
    * production model     = PREFERRED_PRODUCTION_MODEL if it is a transformer whose score is within
                             PREFERRED_MODEL_TOLERANCE of the best transformer; otherwise the best transformer.
    """
    compute_selection_scores(results, settings)
    overall = max(results, key=lambda r: r["selection_score"])
    transformers = [r for r in results if r["type"] == "transformer"]
    production, note = None, "No transformer model finished successfully, so nothing was exported."
    if transformers:
        best_tf = max(transformers, key=lambda r: r["selection_score"])
        preferred_name = settings.PREFERRED_PRODUCTION_MODEL.strip().lower()
        preferred = next((r for r in transformers
                          if preferred_name in (r["model"].lower(), r["key"].lower())), None)
        if preferred is not None and preferred["selection_score"] >= best_tf["selection_score"] - settings.PREFERRED_MODEL_TOLERANCE:
            production = preferred
            note = (f"{preferred['model']} is the configured preferred production model and its score "
                    f"({preferred['selection_score']}) is within {settings.PREFERRED_MODEL_TOLERANCE} of the best "
                    f"transformer ({best_tf['model']}, {best_tf['selection_score']}).")
        else:
            production = best_tf
            note = (f"{best_tf['model']} has the best selection score among transformers "
                    f"({best_tf['selection_score']}); the preferred model '{settings.PREFERRED_PRODUCTION_MODEL}' "
                    "was not available or scored too far below it.")
    for row in results:
        row["overall_winner"] = row is overall
        row["selected_production"] = production is not None and row is production
    return {
        "weights": {"weighted_f1": settings.WEIGHT_F1, "accuracy": settings.WEIGHT_ACCURACY,
                    "latency_inverse": settings.WEIGHT_LATENCY, "size_inverse": settings.WEIGHT_SIZE},
        "formula": "score = 0.40*norm(F1) + 0.30*norm(Accuracy) + 0.20*norm_inverse(Latency) + 0.10*norm_inverse(Size)",
        "overall_winner": overall["model"], "overall_winner_key": overall["key"],
        "production_model": production["model"] if production else None,
        "production_key": production["key"] if production else None,
        "preferred_production_model": settings.PREFERRED_PRODUCTION_MODEL,
        "production_rationale": note,
    }


# --------------------------------------------------------------------------- #
# Output: table, files, report, export
# --------------------------------------------------------------------------- #
def print_benchmark_table(results: List[Dict[str, Any]]) -> None:
    """Print the comparison table."""
    line = "=" * 118
    print("\n" + line)
    print(f"{'MODEL':<22}{'ACCURACY':>10}{'WEIGHTED F1':>14}{'LATENCY(ms)':>14}{'SIZE(MB)':>11}"
          f"{'TRAIN TIME':>13}{'PARAMETERS':>15}{'SCORE':>8}")
    print(line)
    for r in results:
        flag = " <- production" if r.get("selected_production") else ""
        print(f"{r['model']:<22}{r['accuracy']:>10.4f}{r['weighted_f1']:>14.4f}{r['latency_ms']:>14.2f}"
              f"{r['size_mb']:>11.2f}{r['train_time_s']:>12.1f}s{r['parameters']:>15,d}"
              f"{r['selection_score']:>8.3f}{flag}")
    print(line)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def build_report(results: List[Dict[str, Any]], selection: Dict[str, Any], summary: Dict[str, Any],
                 meta: Dict[str, Any], failures: List[str]) -> str:
    """Markdown benchmark report."""
    table = pd.DataFrame([{
        "Model": r["model"], "Accuracy": f"{r['accuracy']:.4f}", "Weighted F1": f"{r['weighted_f1']:.4f}",
        "Precision": f"{r['precision']:.4f}", "Recall": f"{r['recall']:.4f}",
        "Latency (ms/query)": f"{r['latency_ms']:.2f}", "Size (MB)": f"{r['size_mb']:.2f}",
        "Train time (s)": f"{r['train_time_s']:.1f}", "Parameters": f"{r['parameters']:,}",
        "Score": f"{r['selection_score']:.3f}"} for r in results])
    by_latency = sorted(results, key=lambda r: r["latency_ms"])
    by_size = sorted(results, key=lambda r: r["size_mb"])
    lines = [
        "# Benchmark Report", "",
        f"_Generated {meta['created_at']} - mode: **{meta['mode']}** - device: **{meta['device'].upper()}**_", "",
        "All values below were measured by `benchmark_6_models.py` on this machine. Latency depends on the "
        "hardware and device, so compare models only within the same report.", "",
        "## 1. Dataset", f"- Dataset: `{summary['dataset']}`",
        f"- Text column: `{summary['text_column']}`  |  Label column: `{summary['label_column']}`",
        f"- Samples after cleaning: {summary['n_samples_total']} (used in this run: {summary['n_samples_used']})",
        f"- Number of classes: {summary['n_classes']}", "",
        "## 2. Train/test split",
        f"- {summary['n_train']} training / {summary['n_test']} test samples "
        f"(test_size={summary['test_size']}, random_state={summary['random_state']}, stratified where possible)",
        "- The split is made before any TF-IDF fitting or tokenisation; all models use the identical split.", "",
        "## 3. Models and metrics", "Six approaches were attempted; the table lists those that finished.",
        "", _markdown_table(table), "",
    ]
    if failures:
        lines += ["**Models that failed in this run:**", *[f"- {f}" for f in failures], ""]
    lines += [
        "## 4. Selection methodology", f"`{selection['formula']}`", "",
        "Each term is min-max normalised across the models in this run (latency and size are inverted so smaller "
        "is better). Weights: " + ", ".join(f"{k}={v}" for k, v in selection["weights"].items()) + ".", "",
        "## 5. Best model",
        f"- Overall highest selection score: **{selection['overall_winner']}**",
        f"- Best by weighted F1: **{max(results, key=lambda r: r['weighted_f1'])['model']}**", "",
        "## 6. Latency comparison (fastest first)",
        *[f"- {r['model']}: {r['latency_ms']:.2f} ms/query" for r in by_latency], "",
        "## 7. Model size comparison (smallest first)",
        *[f"- {r['model']}: {r['size_mb']:.2f} MB" for r in by_size], "",
        "## 8. Final production model",
        f"- **{selection['production_model'] or 'none'}**", f"- {selection['production_rationale']}", "",
        "The production model is restricted to transformers (a compact SLM); the overall winner is reported "
        "separately and honestly above.", ""]
    return "\n".join(lines)


def save_results(results: List[Dict[str, Any]], selection: Dict[str, Any], summary: Dict[str, Any],
                 meta: Dict[str, Any], failures: List[str]) -> None:
    """Persist CSV, JSON and the Markdown report (previous results are backed up first)."""
    for path in (BENCHMARK_CSV_PATH, BENCHMARK_JSON_PATH):
        if path.exists():
            shutil.copy2(path, path.with_name(path.stem + ".prev" + path.suffix))
    columns = ["model", "accuracy", "weighted_f1", "precision", "recall", "latency_ms", "size_mb",
               "train_time_s", "parameters", "selection_score", "selected_production", "overall_winner",
               "type", "device", "mode", "epochs", "n_train", "n_test", "latency_samples", "hf_name", "key"]
    df = pd.DataFrame(results)[columns]
    _atomic_write(BENCHMARK_CSV_PATH, df.to_csv(index=False))
    payload = {"metadata": {**meta, "dataset": summary}, "selection": selection, "failures": failures,
               "results": results}
    _atomic_write(BENCHMARK_JSON_PATH, json.dumps(payload, indent=2, default=str))
    _atomic_write(BENCHMARK_REPORT_PATH, build_report(results, selection, summary, meta, failures))


def export_production_model(row: Dict[str, Any], bundle: DataBundle, selection: Dict[str, Any],
                            settings: Settings, mode: str, force: bool) -> Optional[Path]:
    """Copy the chosen transformer into ``fine_tuned_intent_model/`` (never leaves a half-written model)."""
    target = settings.model_dir
    metadata_file = target / "training_metadata.json"
    if metadata_file.exists() and not force:
        try:
            existing = json.loads(metadata_file.read_text(encoding="utf-8"))
            if existing.get("mode") == "full" and mode == "quick":
                print("NOTE: a model trained in FULL mode already exists in "
                      f"{target.name}/ - not overwriting it with a quick-mode model "
                      "(use --force-export to override).")
                return None
        except (OSError, ValueError):
            pass
    source = BENCHMARK_MODELS_DIR / row["key"]
    if not (source / "config.json").exists():
        print(f"WARNING: {source} is incomplete; production model not exported.")
        return None

    temp = target.with_name(target.name + "_tmp")
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(source, temp)
    save_label_maps(temp, bundle.label2id, bundle.id2label)
    learning_rate = settings.BERT_TINY_LEARNING_RATE if row["key"] == "bert_tiny" else settings.LEARNING_RATE
    (temp / "training_metadata.json").write_text(json.dumps({
        "model": row["model"], "base_model": row["hf_name"], "benchmark_key": row["key"], "mode": mode,
        "epochs": row["epochs"], "batch_size": settings.BATCH_SIZE, "max_length": settings.MAX_LENGTH,
        "learning_rate": learning_rate, "random_seed": settings.RANDOM_SEED, "device": row["device"],
        "num_labels": len(bundle.label2id), "n_train": row["n_train"], "n_test": row["n_test"],
        "metrics": {k: row[k] for k in ("accuracy", "weighted_f1", "precision", "recall", "latency_ms", "size_mb")},
        "selection_score": row["selection_score"], "production_rationale": selection["production_rationale"],
        "dataset": settings.DATASET_NAME, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    # Swap only after the new copy is complete.
    if target.exists():
        shutil.rmtree(target)
    temp.rename(target)
    return target


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 6 NLP intent classifiers.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="small stratified subset, few epochs (default)")
    mode.add_argument("--full", action="store_true", help="full dataset, TRAINING_EPOCHS epochs")
    parser.add_argument("--models", nargs="+", metavar="KEY",
                        help="only run these models (keys: logistic_regression linear_svm distilbert "
                             "bert_tiny mobilebert roberta_base)")
    parser.add_argument("--resume", action="store_true", help="resume transformer training from checkpoints")
    parser.add_argument("--force-export", action="store_true",
                        help="overwrite a full-mode production model with a quick-mode one")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark. Returns a process exit code."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging()
    ensure_directories(settings)
    quick = True if args.quick else (False if args.full else settings.QUICK_MODE)
    mode = "quick" if quick else "full"
    device = detect_device()
    print(f"Device: {device.upper()}")
    print(f"Mode: {mode.upper()}  (seed={settings.RANDOM_SEED})")
    seed_everything(settings.RANDOM_SEED)

    try:
        bundle = prepare_data(quick, settings)
    except DatasetLoadError as exc:
        print(f"\nERROR: {exc}")
        return 1

    save_label_maps(BENCHMARK_MODELS_DIR, bundle.label2id, bundle.id2label)
    specs = build_model_specs(settings)
    if args.models:
        wanted = {m.lower() for m in args.models}
        specs = [s for s in specs if s.key in wanted or s.name.lower() in wanted]
        if not specs:
            print(f"ERROR: no model matches {args.models}")
            return 1

    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    for number, spec in enumerate(specs, start=1):
        print(f"\n{'-' * 100}\n[{number}/{len(specs)}] {spec.name}\n{'-' * 100}")
        seed_everything(settings.RANDOM_SEED)
        try:
            row = (train_classical(spec, bundle, settings, mode) if spec.kind == "classical"
                   else train_transformer(spec, bundle, settings, device, mode, args.resume))
            results.append(row)
            print(f"Done: accuracy={row['accuracy']:.4f}  weighted_f1={row['weighted_f1']:.4f}  "
                  f"latency={row['latency_ms']:.2f} ms  train_time={row['train_time_s']:.1f}s")
        except ImportError as exc:
            msg = f"{spec.name}: missing package ({exc}). Run `pip install -r requirements.txt`."
            failures.append(msg)
            print(f"ERROR: {msg}")
        except Exception as exc:  # noqa: BLE001 - keep benchmarking the remaining models
            hint = (f" Could not download/load '{spec.hf_name}'. Check your internet connection and retry "
                    "(downloaded models are cached)." if isinstance(exc, OSError) else "")
            msg = f"{spec.name}: {type(exc).__name__}: {exc}.{hint}"
            failures.append(msg)
            logger.debug(traceback.format_exc())
            print(f"ERROR: {msg}")
            gc.collect()

    if not results:
        print("\nERROR: no model finished successfully. Existing results were left untouched.")
        return 1

    selection = choose_models(results, settings)
    print_benchmark_table(results)
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode,
            "device": device, "random_seed": settings.RANDOM_SEED}
    save_results(results, selection, bundle.summary, meta, failures)

    print(f"\nOverall best selection score : {selection['overall_winner']}")
    print(f"Production model (SLM)       : {selection['production_model']}")
    print(f"Why                          : {selection['production_rationale']}")
    if selection["production_key"]:
        row = next(r for r in results if r["key"] == selection["production_key"])
        exported = export_production_model(row, bundle, selection, settings, mode, args.force_export)
        if exported:
            print(f"Production model exported to : {exported}")
    if failures:
        print("\nSome models failed:")
        for item in failures:
            print(f"  - {item}")
    print(f"\nResults saved in {BENCHMARK_CSV_PATH.parent}")
    print("Next: python -m uvicorn main:app --reload   and   streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
