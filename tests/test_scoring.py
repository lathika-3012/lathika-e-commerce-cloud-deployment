import pytest
from pydantic import ValidationError

from schemas import CustomerProfile, IntentResult
from scoring_engine import (calculate_purchase_probability, classify_segment, compute_expected_profit,
                            get_intent_metrics, normalize_intent_label, score_customer)


def profile(cart=100.0, tier="BRONZE", views=0):
    return CustomerProfile(cart_value=cart, tier=tier, page_views_last_10m=views)


def test_vip_beats_bronze():
    assert calculate_purchase_probability(profile(500, "VIP", 5)) > calculate_purchase_probability(profile(500, "BRONZE", 5))


def test_high_cart_beats_low_cart():
    assert calculate_purchase_probability(profile(2000)) > calculate_purchase_probability(profile(10))


def test_zero_page_views_is_valid_and_lower():
    assert 0.0 <= calculate_purchase_probability(profile(100, "GOLD", 0)) < calculate_purchase_probability(profile(100, "GOLD", 15))


def test_probability_is_capped_at_one():
    assert calculate_purchase_probability(profile(1_000_000, "VIP", 10_000)) <= 1.0


def test_segments():
    assert classify_segment(0.80) == "HOT"
    assert classify_segment(0.79) == "WARM"
    assert classify_segment(0.40) == "WARM"
    assert classify_segment(0.39) == "COLD"


def test_expected_profit_formula():
    assert compute_expected_profit(0.5, 100, 0.02, 0.1) == pytest.approx(45.98)


def test_expected_profit_casts_numeric_strings():
    assert compute_expected_profit("0.5", "100", "0.02", "0.1") == pytest.approx(45.98)


def test_expected_profit_rejects_profile_object():
    with pytest.raises(TypeError):
        compute_expected_profit(0.5, profile(), 0.02, 0.1)


def test_profile_validation():
    with pytest.raises(ValidationError):
        profile(cart=-1)
    with pytest.raises(ValidationError):
        profile(views=-3)
    with pytest.raises(ValidationError):
        profile(tier="PLATINUM")
    assert profile(tier=" gold ").tier == "GOLD"


@pytest.mark.parametrize("label,expected", [
    ("payment_issue", "payment_issue"), ("get_refund", "returns_refunds"), ("track_refund", "returns_refunds"),
    ("check_payment_methods", "payment_issue"), ("cancel_order", "order_status"),
    ("contact_customer_service", "general_inquiry"), ("recover_password", None)])
def test_label_normalisation(label, expected):
    assert normalize_intent_label(label) == expected


def test_unknown_intent_falls_back_safely():
    metrics = get_intent_metrics("totally_new_intent")
    assert metrics["risk"] == 0.20 and metrics["revenue"] == "MEDIUM"


def test_score_customer_combines_intent_and_profile():
    result = score_customer(IntentResult(intent="payment_issue", confidence=0.9), profile(100, "GOLD", 5))
    assert result.risk_score == 0.85 and result.revenue_weight == "HIGH"
    assert 0.0 <= result.purchase_probability <= 1.0
