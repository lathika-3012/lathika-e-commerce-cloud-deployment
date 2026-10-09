"""Central configuration for the ecommerce-router project.

Every tunable value lives here. Any value can be overridden with an environment
variable or a ``.env`` file (see ``.env.example``), for example::

    set PROFIT_THRESHOLD=10        (Windows cmd)
    $env:PROFIT_THRESHOLD = "10"   (PowerShell)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Fixed project paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
BENCHMARK_MODELS_DIR: Path = MODELS_DIR / "benchmark_models"
RESULTS_DIR: Path = BASE_DIR / "results"
CONFUSION_MATRIX_DIR: Path = RESULTS_DIR / "confusion_matrices"
CLASSIFICATION_REPORT_DIR: Path = RESULTS_DIR / "classification_reports"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"

BENCHMARK_CSV_PATH: Path = RESULTS_DIR / "benchmark_results.csv"
BENCHMARK_JSON_PATH: Path = RESULTS_DIR / "benchmark_results.json"
BENCHMARK_REPORT_PATH: Path = RESULTS_DIR / "benchmark_report.md"
TELEMETRY_LOG_PATH: Path = RESULTS_DIR / "telemetry.jsonl"
SAMPLE_REQUESTS_PATH: Path = DATA_PROCESSED_DIR / "sample_requests.json"


class Settings(BaseSettings):
    """Runtime settings (environment variables override these defaults)."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Dataset ---------------------------------------------------------- #
    DATASET_NAME: str = "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
    TEST_SIZE: float = 0.2

    # ---- Production intent model ----------------------------------------- #
    MODEL_PATH: str = "./fine_tuned_intent_model"
    FALLBACK_MODEL: str = "distilbert-base-uncased"
    PREFERRED_PRODUCTION_MODEL: str = "DistilBERT"
    # Preferred model is chosen if its selection score is within this margin of
    # the best transformer's score (scores are in the range 0..1).
    PREFERRED_MODEL_TOLERANCE: float = 0.05

    # ---- Simulated model costs (demonstration values, NOT provider prices) -- #
    SLM_COST: float = 0.001
    LLM_COST: float = 0.020

    # ---- Routing thresholds ---------------------------------------------- #
    PROFIT_THRESHOLD: float = 5.0
    HIGH_RISK_THRESHOLD: float = 0.7

    # ---- Training --------------------------------------------------------- #
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 8
    EVAL_BATCH_SIZE: int = 32
    TRAINING_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    BERT_TINY_LEARNING_RATE: float = 1e-4  # tiny models need a larger LR
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    RANDOM_SEED: int = 42

    # ---- Quick mode (beginner-friendly smoke run) ------------------------- #
    QUICK_MODE: bool = True
    QUICK_SAMPLE_SIZE: int = 1500
    QUICK_EPOCHS: int = 1

    # ---- Latency benchmark ------------------------------------------------ #
    LATENCY_SAMPLES: int = 50
    LATENCY_WARMUP: int = 5

    # ---- Model selection weights ----------------------------------------- #
    WEIGHT_F1: float = 0.40
    WEIGHT_ACCURACY: float = 0.30
    WEIGHT_LATENCY: float = 0.20
    WEIGHT_SIZE: float = 0.10

    # ---- LLM provider ("mock" works offline with no API key) -------------- #
    LLM_PROVIDER: str = "mock"  # mock | ollama | openai | groq
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""  # empty -> provider default
    LLM_MODEL_NAME: str = "llama3.2"
    LLM_TIMEOUT_SECONDS: float = 20.0

    # ---- Misc ------------------------------------------------------------- #
    API_URL: str = "http://127.0.0.1:8000"
    TELEMETRY_PERSIST: bool = True
    LOG_LEVEL: str = "INFO"

    @property
    def model_dir(self) -> Path:
        """Absolute path of the fine-tuned production intent model."""
        path = Path(self.MODEL_PATH)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) application settings."""
    return Settings()


def ensure_directories(settings: Optional[Settings] = None) -> None:
    """Create every directory the project writes to."""
    settings = settings or get_settings()
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        BENCHMARK_MODELS_DIR,
        RESULTS_DIR,
        CONFUSION_MATRIX_DIR,
        CLASSIFICATION_REPORT_DIR,
        SCREENSHOTS_DIR,
        settings.model_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def detect_device() -> str:
    """Return ``"cuda"`` if a CUDA GPU is usable, otherwise ``"cpu"``.

    Never raises: if PyTorch is missing or broken the answer is simply "cpu".
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - any failure means "use CPU"
        return "cpu"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure console logging once for the whole application."""
    level_name = (level or get_settings().LOG_LEVEL).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
