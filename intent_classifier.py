"""Layer 2 - Intent intelligence.

Loads the fine-tuned transformer produced by ``benchmark_6_models.py``.

The production flow uses:

1. Explicit business-rule detection for highly unambiguous cases.
2. Fine-tuned transformer (DistilBERT) for general intent classification.
3. Keyword fallback only when the transformer is unavailable.

The explicit business-rule layer prevents obvious high-risk or high-value
queries from being incorrectly mapped to unrelated dataset intents.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

from config import Settings, detect_device, get_settings
from schemas import IntentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

# Used only when the fine-tuned transformer cannot be loaded.

_KEYWORDS: Dict[str, tuple] = {
    "payment_issue": (
        "payment",
        "pay ",
        "paid",
        "charge",
        "charged",
        "charged twice",
        "charged multiple",
        "card",
        "billing",
        "invoice",
        "declined",
        "transaction",
        "checkout failed",
        "payment failed",
    ),

    "returns_refunds": (
        "return",
        "refund",
        "money back",
        "exchange",
        "send back",
        "replace",
        "replacement",
    ),

    "product_recommendation": (
        "recommend",
        "suggest",
        "buy",
        "purchase",
        "looking for",
        "best",
        "which",
        "laptop",
        "phone",
        "headphone",
        "shoes",
        "gift",
        "want a",
        "want to buy",
        "want to purchase",
        "need a",
        "need to buy",
        "gaming",
    ),

    "order_status": (
        "where is my order",
        "order status",
        "track",
        "delivery",
        "shipping",
        "shipped",
        "arrive",
        "my order",
        "package",
        "parcel",
    ),

    "general_inquiry": (
        "hello",
        "hi ",
        "help",
        "opening hours",
        "contact",
        "support",
        "question",
        "information",
        "thanks",
    ),
}


# ---------------------------------------------------------------------------
# Explicit business-intent rules
# ---------------------------------------------------------------------------

# These rules are intentionally limited to highly explicit phrases.
#
# They do NOT replace DistilBERT.
#
# Their purpose is to protect important business intents when the trained
# model predicts an unrelated dataset label.

_BUSINESS_RULES = (

    # ---------------------------------------------------------------
    # Payment / duplicate-charge issues
    # ---------------------------------------------------------------
    (
        "payment_issue",
        (
            "charged twice",
            "charged two times",
            "charged multiple times",
            "charged more than once",
            "charged again",
            "double charged",
            "duplicate charge",
            "duplicate payment",
            "payment failed",
            "payment failure",
            "card was charged",
            "money was deducted twice",
            "money deducted twice",
            "wrong amount charged",
            "incorrect amount charged",
        ),
    ),

    # ---------------------------------------------------------------
    # Returns / exchanges / refunds
    # ---------------------------------------------------------------
    (
        "returns_refunds",
        (
            "can i exchange",
            "can i return",
            "i want to exchange",
            "i want to return",
            "want to exchange",
            "want to return",
            "exchange this product",
            "return this product",
            "return the product",
            "return my order",
            "get a refund",
            "want a refund",
            "request a refund",
            "money back",
        ),
    ),

    # ---------------------------------------------------------------
    # Product purchase / shopping
    # ---------------------------------------------------------------
    (
        "product_recommendation",
        (
            "i want to buy",
            "i want to purchase",
            "want to buy",
            "want to purchase",
            "i want a ",
            "i need a ",
            "i need to buy",
            "i need to purchase",
            "looking for",
            "looking to buy",
            "looking to purchase",
            "want a ",
            "need a ",
            "need to buy",
            "need to purchase",
            "recommend a",
            "recommend me",
            "suggest a",
            "suggest me",
            "which laptop",
            "which phone",
            "which mobile",
            "which computer",
            "which tablet",
            "which product",
            "best laptop",
            "best phone",
            "best mobile",
            "best computer",
            "best tablet",
            "best product",
            "laptop for gaming",
            "gaming laptop",
            "gaming pc",
            "gaming computer",
            "phone for gaming",
            "computer for gaming",
            "pc for gaming",
        ),
    ),
)


def _normalize_text(query: str) -> str:
    """Normalize whitespace and lowercase text for rule matching."""
    return re.sub(
        r"\s+",
        " ",
        query.strip().lower(),
    )


def _business_rule_classify(query: str) -> Optional[IntentResult]:
    """
    Detect highly explicit business intents.

    Returns:
        IntentResult when an explicit rule matches.
        None when the query should be handled by DistilBERT.
    """

    text = _normalize_text(query)

    for intent, phrases in _BUSINESS_RULES:
        for phrase in phrases:
            if phrase in text:
                logger.info(
                    "Business rule matched: query=%r -> intent=%s",
                    query,
                    intent,
                )

                return IntentResult(
                    intent=intent,
                    confidence=0.99,
                    source="business_rule",
                )

    return None


def _keyword_classify(query: str) -> IntentResult:
    """Very small rule-based classifier used only as fallback."""

    text = f" {query.lower()} "

    scores = {
        intent: sum(
            1
            for keyword in keywords
            if keyword in text
        )
        for intent, keywords in _KEYWORDS.items()
    }

    best_intent, best_score = max(
        scores.items(),
        key=lambda item: item[1],
    )

    if best_score == 0:
        return IntentResult(
            intent="general_inquiry",
            confidence=0.30,
            source="keyword",
        )

    confidence = min(
        0.5 + 0.1 * best_score,
        0.9,
    )

    return IntentResult(
        intent=best_intent,
        confidence=confidence,
        source="keyword",
    )


class IntentClassifier:
    """Classify a customer query into an intent with a confidence score."""

    MAX_QUERY_CHARS = 2000

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ) -> None:

        self.settings = settings or get_settings()

        self.model_path = (
            Path(model_path)
            if model_path is not None
            else self.settings.model_dir
        )

        self.device = detect_device()
        self.max_length = self.settings.MAX_LENGTH

        self.backend = "keyword"

        self.tokenizer = None
        self.model = None
        self._torch = None

        self.id2label: Dict[int, str] = {}

        self._load_transformer()

        logger.info(
            "IntentClassifier ready (backend=%s, device=%s)",
            self.backend,
            self.device,
        )

    # ------------------------------------------------------------------
    @property
    def is_transformer(self) -> bool:
        """Return True when the fine-tuned transformer is loaded."""
        return self.backend == "transformer"

    # ------------------------------------------------------------------
    def _load_transformer(self) -> None:
        """
        Try to load the fine-tuned transformer.

        If loading fails, the application falls back to the keyword
        classifier instead of crashing.
        """

        if not (self.model_path / "config.json").exists():
            logger.warning(
                "No fine-tuned model found at %s. "
                "Using keyword fallback. "
                "Run `python benchmark_6_models.py --quick` "
                "to train one.",
                self.model_path,
            )
            return

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path)
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )

            model.to(self.device)
            model.eval()

            id2label = self._read_id2label(model)

        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not load fine-tuned model; "
                "using keyword fallback."
            )
            return

        self._torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.id2label = id2label

        self.backend = "transformer"

    # ------------------------------------------------------------------
    def _read_id2label(self, model) -> Dict[int, str]:
        """
        Prefer ``id2label.json`` saved by the benchmark.

        Otherwise use the model configuration.
        """

        mapping_file = self.model_path / "id2label.json"

        num_labels = int(
            model.config.num_labels
        )

        if mapping_file.exists():
            try:
                raw = json.loads(
                    mapping_file.read_text(
                        encoding="utf-8"
                    )
                )

                mapping = {
                    int(key): str(value)
                    for key, value in raw.items()
                }

                if len(mapping) == num_labels:
                    return mapping

                logger.warning(
                    "id2label.json size mismatch; "
                    "using model config labels."
                )

            except (OSError, ValueError):
                logger.warning(
                    "Could not parse id2label.json; "
                    "using model config labels."
                )

        return {
            int(key): str(value)
            for key, value in model.config.id2label.items()
        }

    # ------------------------------------------------------------------
    def _transformer_classify(
        self,
        query: str,
    ) -> IntentResult:
        """Run the fine-tuned transformer."""

        torch = self._torch

        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = self.model(**encoded).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        confidence, index = torch.max(
            probabilities,
            dim=0,
        )

        label = self.id2label.get(
            int(index),
            str(int(index)),
        )

        confidence_value = float(
            min(
                max(
                    confidence.item(),
                    0.0,
                ),
                1.0,
            )
        )

        return IntentResult(
            intent=label,
            confidence=confidence_value,
            source="transformer",
        )

    # ------------------------------------------------------------------
    def classify(self, query: str) -> IntentResult:
        """
        Classify a customer query.

        Processing order:

        1. Validate and normalize query.
        2. Check explicit business rules.
        3. Use DistilBERT for general queries.
        4. Use keyword fallback if transformer unavailable.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        query = re.sub(
            r"\s+",
            " ",
            query.strip(),
        )[: self.MAX_QUERY_CHARS]

        # ---------------------------------------------------------------
        # Step 1: Explicit business rules
        # ---------------------------------------------------------------

        business_result = _business_rule_classify(
            query
        )

        if business_result is not None:
            return business_result

        # ---------------------------------------------------------------
        # Step 2: Fine-tuned DistilBERT
        # ---------------------------------------------------------------

        if self.is_transformer:
            return self._transformer_classify(
                query
            )

        # ---------------------------------------------------------------
        # Step 3: Keyword fallback
        # ---------------------------------------------------------------

        return _keyword_classify(query)