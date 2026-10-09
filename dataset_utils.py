"""Dataset loading, inspection, label preparation and the shared train/test split."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, Settings, get_settings

logger = logging.getLogger(__name__)

# Column-name hints, tried in order. Real detection never *requires* these names:
# if none match, a heuristic based on the column contents is used.
TEXT_CANDIDATES = ["instruction", "utterance", "customer_query", "user_query", "query", "text",
                   "question", "prompt", "input", "sentence", "message"]
LABEL_CANDIDATES = ["intent", "label", "labels", "category", "class", "tag", "topic"]
NOT_A_QUERY = {"response", "answer", "output", "reply", "completion", "flags"}
MIN_SAMPLES_PER_CLASS = 2  # needed for a stratified split


class DatasetLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded or understood."""


@dataclass
class DataBundle:
    """Everything the benchmark needs: identical split + label maps + summary."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    summary: Dict[str, object]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw_dataset(dataset_name: str) -> pd.DataFrame:
    """Load every split of the Hugging Face dataset into one DataFrame.

    A copy is cached in ``data/raw`` so later runs still work offline.
    """
    cache_file = DATA_RAW_DIR / "bitext_raw.csv"
    try:
        from datasets import load_dataset

        print(f"Downloading/loading dataset '{dataset_name}' (needs internet on the first run)...")
        dataset = load_dataset(dataset_name)
        print("\n=== DATASET STRUCTURE ===")
        print(dataset)
        frames: List[pd.DataFrame] = []
        for split_name, split in dataset.items():
            frame = split.to_pandas()
            for column, feature in split.features.items():  # turn ClassLabel ids into names
                names = getattr(feature, "names", None)
                if names and column in frame.columns:
                    frame[column] = frame[column].map(
                        lambda i, n=names: n[i] if isinstance(i, (int, np.integer)) and 0 <= i < len(n) else i)
            logger.info("Split '%s': %d rows, columns=%s", split_name, len(frame), list(frame.columns))
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as exc:  # noqa: BLE001
        if cache_file.exists():
            print(f"WARNING: could not load '{dataset_name}' ({type(exc).__name__}). "
                  f"Using cached copy {cache_file}.")
            return pd.read_csv(cache_file)
        raise DatasetLoadError(
            f"Could not load dataset '{dataset_name}': {exc}\n"
            "Check your internet connection (the dataset downloads from Hugging Face on the first run) "
            "and that the 'datasets' package is installed.") from exc


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def _string_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def detect_text_column(df: pd.DataFrame) -> str:
    """Find the customer-query column."""
    lower = {str(c).lower(): c for c in df.columns}
    strings = set(_string_columns(df))
    for candidate in TEXT_CANDIDATES:
        if candidate in lower and lower[candidate] in strings:
            return lower[candidate]
    best, best_len = None, -1.0
    for column in strings:
        if str(column).lower() in NOT_A_QUERY or str(column).lower() in LABEL_CANDIDATES:
            continue
        series = df[column].dropna().astype(str)
        if series.empty or series.nunique() / len(series) < 0.3:
            continue  # low-cardinality text is a label, not a query
        mean_len = series.str.len().mean()
        if mean_len > best_len:
            best, best_len = column, mean_len
    if best is None:
        raise DatasetLoadError(f"Could not detect a text column. Columns: {list(df.columns)}")
    return best


def detect_label_column(df: pd.DataFrame, text_column: str) -> str:
    """Find the intent/category column."""
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lower and lower[candidate] != text_column:
            return lower[candidate]
    best, best_unique = None, -1
    for column in df.columns:
        if column == text_column or str(column).lower() in NOT_A_QUERY:
            continue
        unique = df[column].nunique(dropna=True)
        if 2 <= unique <= min(500, max(len(df) // 5, 2)) and unique > best_unique:
            best, best_unique = column, unique
    if best is None:
        raise DatasetLoadError(f"Could not detect a label column. Columns: {list(df.columns)}")
    return best


# --------------------------------------------------------------------------- #
# Cleaning, summary, label maps
# --------------------------------------------------------------------------- #
def normalize_dataframe(df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    """Return a clean frame with exactly two columns: ``text`` and ``label``."""
    valid = df[text_column].notna() & df[label_column].notna()
    out = pd.DataFrame({"text": df.loc[valid, text_column].astype(str).str.strip(),
                        "label": df.loc[valid, label_column].astype(str).str.strip()})
    out = out[(out["text"] != "") & (out["label"] != "") & (out["label"].str.lower() != "nan")]
    before = len(out)
    out = out.drop_duplicates(subset="text").reset_index(drop=True)  # identical texts could leak across the split
    if before != len(out):
        print(f"Removed {before - len(out)} duplicate query texts (prevents train/test leakage).")
    counts = out["label"].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_CLASS].index
    if len(rare):
        print(f"WARNING: dropping {len(rare)} class(es) with fewer than {MIN_SAMPLES_PER_CLASS} samples: {list(rare)}")
        out = out[~out["label"].isin(rare)].reset_index(drop=True)
    return out


def print_dataset_summary(df: pd.DataFrame, text_column: str, label_column: str) -> None:
    """Print sample count, classes, distribution and examples."""
    print("\n=== DATASET SUMMARY ===")
    print(f"Detected text column : {text_column}")
    print(f"Detected label column: {label_column}")
    print(f"Number of samples    : {len(df)}")
    print(f"Number of classes    : {df['label'].nunique()}")
    print(f"Class names          : {sorted(df['label'].unique())}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print("\nExample samples:")
    for _, row in df.sample(n=min(5, len(df)), random_state=0).iterrows():
        print(f"  [{row['label']}] {row['text'][:110]}")
    print()


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create deterministic label<->id maps from any set of labels."""
    classes = sorted({str(label) for label in labels})
    label2id = {label: idx for idx, label in enumerate(classes)}
    return label2id, {idx: label for label, idx in label2id.items()}


def save_label_maps(directory: Path, label2id: Dict[str, int], id2label: Dict[int, str]) -> None:
    """Write ``label2id.json`` and ``id2label.json`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "label2id.json").write_text(json.dumps(label2id, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "id2label.json").write_text(
        json.dumps({str(k): v for k, v in id2label.items()}, indent=2, ensure_ascii=False), encoding="utf-8")


def load_label_maps(directory: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read the maps written by :func:`save_label_maps`."""
    label2id = json.loads((directory / "label2id.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in json.loads((directory / "id2label.json").read_text(encoding="utf-8")).items()}
    return label2id, id2label


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    except ValueError as exc:
        print(f"WARNING: stratified split not possible ({exc}); using a random split instead.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, quick: bool = False,
                  quick_sample_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split (``random_state=42``). Quick mode first draws a stratified subset.

    The split happens BEFORE any vectoriser/tokenizer/model sees the data, and the
    same two frames are handed to all six models.
    """
    work = df
    if quick and len(df) > quick_sample_size:
        try:
            work, _ = train_test_split(df, train_size=quick_sample_size, random_state=seed, stratify=df["label"])
        except ValueError:
            work = df.sample(n=quick_sample_size, random_state=seed)
        work = work.reset_index(drop=True)
        print(f"QUICK MODE: using a stratified subset of {len(work)} samples.")
    return _stratified_split(work, test_size, seed)


def prepare_data(quick: bool, settings: Optional[Settings] = None) -> DataBundle:
    """Full data pipeline: load -> inspect -> normalise -> split -> label maps -> save."""
    settings = settings or get_settings()
    raw = load_raw_dataset(settings.DATASET_NAME)
    text_col = detect_text_column(raw)
    label_col = detect_label_column(raw, text_col)
    clean = normalize_dataframe(raw, text_col, label_col)
    if clean.empty:
        raise DatasetLoadError("No usable rows after cleaning the dataset.")
    print_dataset_summary(clean, text_col, label_col)

    train_df, test_df = split_dataset(clean, settings.TEST_SIZE, settings.RANDOM_SEED, quick,
                                      settings.QUICK_SAMPLE_SIZE)
    label2id, id2label = build_label_maps(list(train_df["label"]) + list(test_df["label"]))

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)
    summary: Dict[str, object] = {
        "dataset": settings.DATASET_NAME, "mode": "quick" if quick else "full",
        "text_column": text_col, "label_column": label_col,
        "n_samples_total": int(len(clean)), "n_samples_used": int(len(train_df) + len(test_df)),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)), "n_classes": len(label2id),
        "class_names": list(label2id), "test_size": settings.TEST_SIZE, "random_state": settings.RANDOM_SEED,
        "class_distribution": {k: int(v) for k, v in clean["label"].value_counts().items()},
    }
    (DATA_PROCESSED_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Train/test split: {len(train_df)} / {len(test_df)} "
          f"(test_size={settings.TEST_SIZE}, random_state={settings.RANDOM_SEED}, stratified where possible)")
    return DataBundle(train_df, test_df, label2id, id2label, summary)
