"""The seven-layer gateway pipeline shared by the FastAPI backend and the dashboard.

Processing order (data dependencies decide the order):

    Layer 1  Request Intake                   validated ChatRequest
    Layer 2  Intent Intelligence              IntentClassifier
    Layer 3  Customer Engagement Scoring      purchase probability, segment, intent risk
    Layer 4  Profit & Risk Decision Engine    expected profit  (inside Router.route)
    Layer 5  Dynamic Model Routing             SLM or LLM       (inside Router.route)
    Layer 6  Response Generation              LLMService
    Layer 7  Telemetry & Financial Monitoring  Telemetry
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from config import Settings, get_settings
from intent_classifier import IntentClassifier
from llm_service import LLMService
from router import Router
from schemas import ChatRequest, ChatResponse
from scoring_engine import score_customer
from telemetry import Telemetry

logger = logging.getLogger(__name__)

LAYERS = (
    "Layer 1 - Request Intake",
    "Layer 2 - Intent Intelligence",
    "Layer 3 - Customer Engagement Scoring",
    "Layer 4 - Profit & Risk Decision Engine",
    "Layer 5 - Dynamic Model Routing (SLM / LLM)",
    "Layer 6 - Response Generation",
    "Layer 7 - Telemetry & Financial Monitoring",
)


def extract_purchase_value(query: str) -> Optional[float]:
    """Extract an explicit purchase amount from a customer query.

    Examples:
        'I want to buy a laptop for 20000' -> 20000.0
        'phone around ₹35,000'              -> 35000.0
        'Where is my order?'                -> None

    The value is query-specific and is therefore kept separate from
    CustomerProfile.cart_value.
    """
    if not isinstance(query, str):
        return None

    text = query.lower()

    # Only look for amounts near purchase/budget/pricing language.
    purchase_context = (
        "buy", "purchase", "price", "cost", "budget",
        "spend", "worth", "around", "under", "below",
        "within", "for"
    )

    if not any(keyword in text for keyword in purchase_context):
        return None

    # Supports:
    #   20000
    #   20,000
    #   ₹20,000
    #   $20,000
    #   20k / 20K
    pattern = re.compile(
        r"(?:₹|\$|rs\.?\s*)?\s*"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*(k|thousand|lakh)?",
        re.IGNORECASE,
    )

    matches = pattern.findall(text)

    if not matches:
        return None

    values = []

    for number_text, suffix in matches:
        try:
            value = float(number_text.replace(",", ""))

            suffix = suffix.lower()

            if suffix == "k" or suffix == "thousand":
                value *= 1_000
            elif suffix == "lakh":
                value *= 100_000

            if value > 0:
                values.append(value)

        except ValueError:
            continue

    if not values:
        return None

    # Use the largest detected purchase-related amount.
    return max(values)


class GatewayPipeline:
    """Runs one customer query through all seven layers."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        classifier: Optional[IntentClassifier] = None,
        router: Optional[Router] = None,
        llm: Optional[LLMService] = None,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.classifier = classifier or IntentClassifier(settings=self.settings)
        self.router = router or Router(self.settings)
        self.llm = llm or LLMService(self.settings)
        self.telemetry = telemetry or Telemetry(self.settings)

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Process a validated request and return the full decision + response."""
        started = time.perf_counter()

        # Layer 1 - Request Intake
        query, profile = request.query, request.profile

        # Layer 2 - Intent Intelligence
        intent_res = self.classifier.classify(query)

        # Layer 3 - Customer Engagement Scoring
        scoring_res = score_customer(intent_res, profile)

        # --------------------------------------------------------------- #
        # Query-aware purchase value
        #
        # If the customer explicitly mentions a purchase amount, use it
        # for the profit calculation. Otherwise preserve the existing
        # profile.cart_value behavior.
        # --------------------------------------------------------------- #
        query_purchase_value = extract_purchase_value(query)

        effective_purchase_value = (
            query_purchase_value
            if query_purchase_value is not None
            else profile.cart_value
        )

        if query_purchase_value is not None:
            logger.info(
                "Query purchase value detected: %.2f | profile cart value: %.2f | "
                "effective purchase value: %.2f",
                query_purchase_value,
                profile.cart_value,
                effective_purchase_value,
            )

        # Layers 4 + 5 - Profit & Risk decision, then SLM/LLM routing
        decision = self.router.route(
            intent_res,
            scoring_res,
            effective_purchase_value,
        )

        # Layer 6 - Response Generation
        text, source = self.llm.respond(
            decision.selected_model,
            query,
            intent_res.intent,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0

        # Layer 7 - Telemetry & Financial Monitoring
        record = self.telemetry.record(
            intent=intent_res.intent,
            segment=scoring_res.segment,
            selected_model=decision.selected_model,
            expected_profit=decision.expected_profit,
            model_cost=decision.model_cost,
            latency_ms=latency_ms,
        )

        return ChatResponse(
            intent=intent_res.intent,
            canonical_intent=scoring_res.canonical_intent,
            confidence=intent_res.confidence,
            classifier_backend=intent_res.source,
            segment=scoring_res.segment,
            purchase_probability=scoring_res.purchase_probability,
            risk_score=scoring_res.risk_score,
            revenue_weight=scoring_res.revenue_weight,
            selected_model=decision.selected_model,
            expected_profit=round(decision.expected_profit, 4),
            model_cost=decision.model_cost,
            estimated_savings=record.estimated_savings,
            reason=decision.reason,
            response=text,
            response_source=source,
            latency=round(latency_ms, 3),
        )