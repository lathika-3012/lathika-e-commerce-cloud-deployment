"""Layers 4 & 5 - Profit & Risk decision engine and dynamic SLM/LLM routing."""

from __future__ import annotations

import logging
from typing import Optional

from config import Settings, get_settings
from schemas import IntentResult, RouteDecision, ScoringResult
from scoring_engine import compute_expected_profit

logger = logging.getLogger(__name__)


class Router:
    """
    Choose between SLM and LLM using intent risk, revenue value,
    and expected profit.

    Routing strategy:

        1. HIGH-RISK intent -> LLM
        2. HIGH-revenue intent + expected profit above threshold -> LLM
        3. Otherwise -> SLM

    Why this strategy?

    Expected profit alone can become extremely large for customers
    with high cart values. If expected profit were the only trigger,
    even LOW-risk requests such as "Where is my order?" would almost
    always be escalated to the expensive LLM.

    Therefore:
        - Risk is used as the primary safety/escalation signal.
        - Expected profit is used for financially valuable intents.
        - Low-risk, low-value requests are handled by the cheaper SLM.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        settings = settings or get_settings()

        self.profit_threshold = float(settings.PROFIT_THRESHOLD)
        self.high_risk_threshold = float(settings.HIGH_RISK_THRESHOLD)

        self.slm_cost = float(settings.SLM_COST)
        self.llm_cost = float(settings.LLM_COST)

    def route(
        self,
        intent_res: IntentResult,
        scoring_res: ScoringResult,
        cart_value: float,
    ) -> RouteDecision:
        """
        Return the routing decision with a human-readable reason.

        Routing rules:

            HIGH RISK
                -> LLM

            HIGH REVENUE + PROFIT ABOVE THRESHOLD
                -> LLM

            Otherwise
                -> SLM
        """

        # --------------------------------------------------------------- #
        # Calculate expected profit for the LLM escalation option.
        # --------------------------------------------------------------- #
        expected_profit = compute_expected_profit(
            scoring_res.purchase_probability,
            float(cart_value),
            self.llm_cost,
            scoring_res.risk_score,
        )

        # --------------------------------------------------------------- #
        # Determine routing triggers.
        # --------------------------------------------------------------- #
        profit_trigger = expected_profit >= self.profit_threshold

        risk_trigger = scoring_res.risk_score >= self.high_risk_threshold

        high_value_intent = scoring_res.revenue_weight.upper() == "HIGH"

        # --------------------------------------------------------------- #
        # Dynamic routing decision.
        # --------------------------------------------------------------- #

        # Rule 1:
        # High-risk requests always go to the more capable LLM.
        if risk_trigger:
            selected = "LLM"

            reason = (
                "LLM selected because intent risk exceeded the "
                "high-risk threshold."
            )

        # Rule 2:
        # High-value intents can justify the expensive LLM when
        # their expected profit exceeds the configured threshold.
        elif profit_trigger and high_value_intent:
            selected = "LLM"

            reason = (
                "LLM selected because expected profit exceeded the "
                "configured threshold for a high-value intent."
            )

        # Rule 3:
        # Everything else uses the cheaper SLM.
        else:
            selected = "SLM"

            reason = (
                "SLM selected because the intent was not high-risk "
                "and did not meet the high-value profit escalation rule."
            )

        # --------------------------------------------------------------- #
        # Select simulated model cost.
        # --------------------------------------------------------------- #
        model_cost = (
            self.llm_cost
            if selected == "LLM"
            else self.slm_cost
        )

        # --------------------------------------------------------------- #
        # Logging for debugging and telemetry.
        # --------------------------------------------------------------- #
        logger.info(
            "Routing decision: "
            "intent=%s | canonical=%s | revenue=%s | "
            "profit=%.2f | profit_trigger=%s | "
            "risk=%.2f | risk_trigger=%s | "
            "selected=%s",
            intent_res.intent,
            scoring_res.canonical_intent,
            scoring_res.revenue_weight,
            expected_profit,
            profit_trigger,
            scoring_res.risk_score,
            risk_trigger,
            selected,
        )

        # --------------------------------------------------------------- #
        # Return the final routing decision.
        # --------------------------------------------------------------- #
        return RouteDecision(
            selected_model=selected,
            expected_profit=float(expected_profit),
            model_cost=model_cost,
            reason=reason,
        )