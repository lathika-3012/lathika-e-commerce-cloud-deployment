"""Layer 3 - Customer engagement & intent-risk scoring, plus the profit formula."""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from schemas import CustomerProfile, IntentResult, ScoringResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Intent risk map (canonical intents)
# --------------------------------------------------------------------------- #
INTENT_METRICS: Dict[str, Dict[str, object]] = {
    "payment_issue": {"risk": 0.85, "revenue": "HIGH"},
    "returns_refunds": {"risk": 0.65, "revenue": "MEDIUM"},
    "product_recommendation": {"risk": 0.10, "revenue": "HIGH"},
    "order_status": {"risk": 0.15, "revenue": "LOW"},
    "general_inquiry": {"risk": 0.05, "revenue": "LOW"},
}

# Unknown intents fall back to this entry (never crash).
DEFAULT_INTENT_METRICS: Dict[str, object] = {"risk": 0.20, "revenue": "MEDIUM"}

# The real Bitext labels differ from the five canonical intents above, so labels
# are mapped with ordered keyword stems. The first matching rule wins, which is
# why refund/return rules come before the generic "order"/"track" rule
# (e.g. "track_refund" -> returns_refunds). Edit this list to tune the mapping.
_MAPPING_RULES = (
    ("returns_refunds", ("refund", "return", "money_back", "exchange")),
    ("payment_issue", ("pay", "invoice", "bill", "charg", "card", "transaction")),
    ("product_recommendation", ("recommend", "suggest", "product", "search", "browse",
                                "review", "compar", "catalog", "item")),
    ("order_status", ("order", "track", "deliver", "ship", "status", "cancel",
                      "package", "parcel")),
    ("general_inquiry", ("contact", "inquir", "faq", "newsletter", "general",
                         "greeting", "thank")),
)

# --------------------------------------------------------------------------- #
# Purchase probability parameters
# --------------------------------------------------------------------------- #
BASE_PROBABILITY = 0.15
CART_MAX_CONTRIBUTION = 0.30
CART_SATURATION_VALUE = 2000.0     # cart value at which the cart bonus is maxed
ENGAGEMENT_MAX_CONTRIBUTION = 0.25
ENGAGEMENT_SATURATION_VIEWS = 20   # page views at which the engagement bonus is maxed
TIER_CONTRIBUTION: Dict[str, float] = {"VIP": 0.20, "GOLD": 0.14, "SILVER": 0.08, "BRONZE": 0.03}

HOT_THRESHOLD = 0.80
WARM_THRESHOLD = 0.40

# --------------------------------------------------------------------------- #
# Profit formula constants
# --------------------------------------------------------------------------- #
MARGIN_RATE = 0.95
RISK_PENALTY_FACTOR = 15.0


def normalize_intent_label(label: str) -> Optional[str]:
    """Map any dataset label to one of the canonical intents, or ``None``."""
    if not isinstance(label, str) or not label.strip():
        return None
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if cleaned in INTENT_METRICS:
        return cleaned
    tokens = [t for t in cleaned.split("_") if t]
    for canonical, stems in _MAPPING_RULES:
        for token in tokens:
            if any(token.startswith(stem) for stem in stems):
                return canonical
    joined = cleaned
    for canonical, stems in _MAPPING_RULES:  # catches stems spanning tokens, e.g. "money_back"
        if any(stem in joined for stem in stems if "_" in stem):
            return canonical
    return None


def get_intent_metrics(label: str) -> Dict[str, object]:
    """Return ``{"canonical", "risk", "revenue"}`` for any label (safe fallback)."""
    canonical = normalize_intent_label(label)
    if canonical is None:
        logger.info("Unknown intent label; using default risk metrics.")
        return {"canonical": "unknown", **DEFAULT_INTENT_METRICS}
    return {"canonical": canonical, **INTENT_METRICS[canonical]}


def calculate_purchase_probability(profile: CustomerProfile) -> float:
    """Estimate purchase probability from cart value, engagement and tier.

    probability = base + cart contribution + engagement contribution + tier contribution,
    clamped to ``[0.0, 1.0]``.
    """
    cart_part = CART_MAX_CONTRIBUTION * min(float(profile.cart_value) / CART_SATURATION_VALUE, 1.0)
    engagement_part = ENGAGEMENT_MAX_CONTRIBUTION * min(
        float(profile.page_views_last_10m) / ENGAGEMENT_SATURATION_VIEWS, 1.0
    )
    tier_part = TIER_CONTRIBUTION.get(str(profile.tier).upper(), 0.0)
    probability = BASE_PROBABILITY + cart_part + engagement_part + tier_part
    return float(min(max(probability, 0.0), 1.0))


def classify_segment(purchase_probability: float) -> str:
    """HOT >= 0.80, WARM >= 0.40, otherwise COLD."""
    p = float(purchase_probability)
    if p >= HOT_THRESHOLD:
        return "HOT"
    if p >= WARM_THRESHOLD:
        return "WARM"
    return "COLD"


def compute_expected_profit(purchase_prob: float, cart_value: float,
                            model_cost: float, risk_score: float) -> float:
    """Expected Profit = (p x cart x 0.95) - model_cost - (risk x 15.0).

    All inputs are explicitly cast to ``float`` so that numeric strings work and
    accidentally passing the wrong object fails with a clear error.
    """
    purchase_prob = float(purchase_prob)
    cart_value = float(cart_value)
    model_cost = float(model_cost)
    risk_score = float(risk_score)
    return (purchase_prob * cart_value * MARGIN_RATE) - model_cost - (risk_score * RISK_PENALTY_FACTOR)


def score_customer(intent_res: IntentResult, profile: CustomerProfile) -> ScoringResult:
    """Combine intent risk with the customer's engagement into a ``ScoringResult``."""
    metrics = get_intent_metrics(intent_res.intent)
    probability = calculate_purchase_probability(profile)
    return ScoringResult(
        segment=classify_segment(probability),
        purchase_probability=probability,
        risk_score=float(metrics["risk"]),
        revenue_weight=str(metrics["revenue"]),
        canonical_intent=str(metrics["canonical"]),
    )
