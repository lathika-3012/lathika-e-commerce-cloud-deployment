from router import Router
from schemas import IntentResult, ScoringResult


INTENT = IntentResult(
    intent="order_status",
    confidence=0.9,
)


def scoring(
    prob,
    risk,
    revenue="LOW",
    canonical="order_status",
):
    return ScoringResult(
        segment="WARM",
        purchase_probability=prob,
        risk_score=risk,
        revenue_weight=revenue,
        canonical_intent=canonical,
    )


def test_high_value_profitable_intent_goes_to_llm():
    """
    A high-revenue intent with high expected profit
    should be routed to the LLM.
    """

    decision = Router().route(
        INTENT,
        scoring(
            0.8,
            0.1,
            revenue="HIGH",
            canonical="product_recommendation",
        ),
        1500,
    )

    assert decision.selected_model == "LLM"
    assert "expected profit" in decision.reason
    assert decision.expected_profit >= 5.0


def test_high_risk_goes_to_llm_even_with_empty_cart():
    """
    High-risk queries should go to the LLM
    even when there is no cart value.
    """

    decision = Router().route(
        INTENT,
        scoring(
            0.2,
            0.85,
            revenue="HIGH",
            canonical="payment_issue",
        ),
        0,
    )

    assert decision.selected_model == "LLM"
    assert "risk" in decision.reason.lower()


def test_low_profit_and_low_risk_goes_to_slm():
    """
    Simple low-risk, low-value requests should use the SLM.
    """

    decision = Router().route(
        INTENT,
        scoring(
            0.2,
            0.05,
            revenue="LOW",
            canonical="general_inquiry",
        ),
        10,
    )

    assert decision.selected_model == "SLM"
    assert decision.reason.startswith("SLM selected")


def test_high_profit_low_revenue_stays_slm():
    """
    High expected profit alone should NOT force an LLM
    when the intent is LOW revenue and LOW risk.

    This is the important behavior introduced by our
    modified routing strategy.
    """

    decision = Router().route(
        INTENT,
        scoring(
            0.8,
            0.1,
            revenue="LOW",
            canonical="order_status",
        ),
        1500,
    )

    assert decision.selected_model == "SLM"
    assert decision.expected_profit >= 5.0
    assert "high-value profit escalation" in decision.reason


def test_low_risk_boundary_just_below_threshold_stays_slm():
    """
    Risk just below the configured high-risk threshold
    should remain on the SLM path when the intent is LOW revenue.
    """

    decision = Router().route(
        INTENT,
        scoring(
            0.1,
            0.69,
            revenue="LOW",
            canonical="order_status",
        ),
        0,
    )

    assert decision.selected_model == "SLM"


def test_model_cost_for_slm():
    """
    SLM routing should use the configured SLM cost.
    """

    router = Router()

    decision = router.route(
        INTENT,
        scoring(
            0.2,
            0.05,
            revenue="LOW",
            canonical="order_status",
        ),
        10,
    )

    assert decision.selected_model == "SLM"
    assert decision.model_cost == router.slm_cost


def test_model_cost_for_llm():
    """
    LLM routing should use the configured LLM cost.
    """

    router = Router()

    decision = router.route(
        INTENT,
        scoring(
            0.8,
            0.1,
            revenue="HIGH",
            canonical="product_recommendation",
        ),
        1500,
    )

    assert decision.selected_model == "LLM"
    assert decision.model_cost == router.llm_cost