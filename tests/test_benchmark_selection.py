import pandas as pd
import pytest

from benchmark_6_models import choose_models
from config import Settings
from dataset_utils import (build_label_maps, detect_label_column, detect_text_column, normalize_dataframe,
                           split_dataset)


def row(name, key, kind, acc, f1, lat, size):
    return {"model": name, "key": key, "type": kind, "accuracy": acc, "weighted_f1": f1,
            "latency_ms": lat, "size_mb": size}


def test_preferred_model_chosen_when_close():
    results = [row("Logistic Regression", "logistic_regression", "classical", 0.90, 0.90, 0.5, 1),
               row("DistilBERT", "distilbert", "transformer", 0.99, 0.99, 20, 250),
               row("BERT Tiny", "bert_tiny", "transformer", 0.97, 0.97, 3, 17),
               row("RoBERTa Base", "roberta_base", "transformer", 0.99, 0.99, 40, 480)]
    selection = choose_models(results, Settings(PREFERRED_MODEL_TOLERANCE=1.0))
    assert selection["production_model"] == "DistilBERT"
    assert all("selection_score" in r for r in results)


def test_actual_winner_reported_when_preferred_is_far_behind():
    results = [row("DistilBERT", "distilbert", "transformer", 0.60, 0.60, 50, 250),
               row("BERT Tiny", "bert_tiny", "transformer", 0.99, 0.99, 2, 17)]
    selection = choose_models(results, Settings(PREFERRED_MODEL_TOLERANCE=0.05))
    assert selection["production_model"] == "BERT Tiny"


def test_no_transformer_means_no_production_model():
    selection = choose_models([row("Linear SVM", "linear_svm", "classical", 0.9, 0.9, 1, 1)], Settings())
    assert selection["production_model"] is None


def test_column_detection_does_not_assume_names():
    df = pd.DataFrame({"flags": ["B", "BQ"] * 50,
                       "customer_message": [f"unique customer question number {i} about orders" for i in range(100)],
                       "response": [f"a long answer to number {i} " * 5 for i in range(100)],
                       "kind": ["a", "b", "c", "d"] * 25})
    text_col = detect_text_column(df)
    assert text_col == "customer_message"
    assert detect_label_column(df, text_col) == "kind"


def test_split_is_deterministic_and_disjoint():
    df = pd.DataFrame({"text": [f"query {i}" for i in range(200)], "label": ["a", "b"] * 100})
    train1, test1 = split_dataset(df, 0.2, 42)
    train2, test2 = split_dataset(df, 0.2, 42)
    assert train1.equals(train2) and test1.equals(test2)
    assert not set(train1["text"]) & set(test1["text"])
    assert len(test1) == 40


def test_normalize_drops_missing_and_duplicates():
    raw = pd.DataFrame({"q": ["a", "a", None, "b", "c", "c2", "d"], "intent": ["x", "x", "x", None, "y", "y", "x"]})
    clean = normalize_dataframe(raw, "q", "intent")
    assert list(clean.columns) == ["text", "label"]
    assert len(clean) == 4  # a, c, c2, d (b has no label; None has no text; second 'a' is a duplicate)


def test_label_maps_roundtrip():
    label2id, id2label = build_label_maps(["b", "a", "b"])
    assert label2id == {"a": 0, "b": 1} and id2label == {0: "a", 1: "b"}
