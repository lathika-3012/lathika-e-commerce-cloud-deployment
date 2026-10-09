# ============================================================
# DYNAMIC REVENUE + RISK SCORING
# E-COMMERCE AI GATEWAY
# ============================================================

"""
Layer 2 calculates dynamic business values for the
CURRENT customer + CURRENT query.

Important:
- These are formula-based business scores.
- They are NOT ML predictions.
- They are NOT claimed to be real-world revenue/risk probabilities.
- The formulas use customer data and Layer-1 signals.
"""


# ============================================================
# INTENT BUSINESS SEVERITY
# ============================================================

# This is NOT the final revenue/risk score.
#
# It represents the inherent business severity of an intent.
#
# We retain domain knowledge here because some intents
# naturally carry more business risk than others.
#
# Example:
# payment_issue is inherently more sensitive than
# store_opening_hours.

INTENT_SEVERITY = {

    # --------------------------------------------------------
    # PRODUCT / SHOPPING
    # --------------------------------------------------------

    "add_product": 0.30,

    "product_information": 0.20,

    "availability": 0.20,

    "availability_in_store": 0.20,

    "availability_online": 0.20,

    "sales_period": 0.20,

    "exchange_product": 0.50,

    "exchange_product_in_store": 0.50,

    "return_product": 0.70,

    "return_product_in_store": 0.70,

    "return_product_online": 0.70,

    "return_policy": 0.50,


    # --------------------------------------------------------
    # ORDERS / DELIVERY
    # --------------------------------------------------------

    "track_order": 0.20,

    "track_delivery": 0.25,

    "delivery_time": 0.20,

    "delivery_issue": 0.60,

    "damaged_delivery": 0.75,

    "missing_item": 0.75,

    "wrong_item": 0.70,

    "shipping_costs": 0.20,

    "store_location": 0.10,

    "store_opening_hours": 0.05,


    # --------------------------------------------------------
    # ORDER MANAGEMENT
    # --------------------------------------------------------

    "cancel_order": 0.65,

    "change_order": 0.50,

    "remove_product": 0.35,

    "order_history": 0.15,

    "request_invoice": 0.20,


    # --------------------------------------------------------
    # PAYMENT / REFUND
    # --------------------------------------------------------

    "pay": 0.50,

    "payment_methods": 0.15,

    "payment_issue": 0.75,

    "request_refund": 0.85,

    "refund_policy": 0.60,

    "refund_status": 0.65,


    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    "open_account": 0.20,

    "close_account": 0.70,

    "change_account": 0.40,

    "recover_password": 0.55,

    "request_right_to_rectification": 0.80,


    # --------------------------------------------------------
    # CUSTOMER SUPPORT
    # --------------------------------------------------------

    "customer_service": 0.40,

    "human_agent": 0.80,

    "submit_feedback": 0.30,

    "submit_product_feedback": 0.25,

    "submit_product_idea": 0.15,


    # --------------------------------------------------------
    # TECHNICAL / PRODUCT ISSUES
    # --------------------------------------------------------

    "technical_issue": 0.60,

    "product_issue": 0.65,
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):
    """
    Keep a score between 0 and 1.
    """

    return max(
        minimum,
        min(maximum, float(value))
    )


def normalize_order_value(
    order_value,
    reference_value=5000.0
):
    """
    Convert order value into a 0-1 business-value signal.

    ₹0      -> 0.00
    ₹2500   -> 0.50
    ₹5000+  -> 1.00

    The reference value is a prototype business-policy
    normalization point, not a learned parameter.
    """

    if order_value is None:
        return 0.0

    order_value = max(
        0.0,
        float(order_value)
    )

    return clamp(
        order_value / reference_value
    )


def normalize_cart_value(
    cart_value,
    reference_value=5000.0
):
    """
    Convert cart value into a 0-1 signal.
    """

    if cart_value is None:
        return 0.0

    cart_value = max(
        0.0,
        float(cart_value)
    )

    return clamp(
        cart_value / reference_value
    )


def normalize_previous_purchases(
    previous_purchases,
    reference_value=10.0
):
    """
    Convert purchase history into a 0-1 signal.
    """

    if previous_purchases is None:
        return 0.0

    previous_purchases = max(
        0.0,
        float(previous_purchases)
    )

    return clamp(
        previous_purchases / reference_value
    )


def normalize_customer_tier(
    customer_tier
):
    """
    Convert customer tier to a 0-1 value.

    0 = New
    1 = Regular
    2 = Premium
    """

    try:

        tier = float(
            customer_tier
        )

    except (
        TypeError,
        ValueError
    ):

        tier = 0.0

    return clamp(
        tier / 2.0
    )


# ============================================================
# INTENT SEVERITY
# ============================================================

def get_intent_severity(
    intent
):
    """
    Return inherent business severity of an intent.

    Unknown/uncertain intents receive a neutral value.
    """

    return float(
        INTENT_SEVERITY.get(
            intent,
            0.50
        )
    )


# ============================================================
# DYNAMIC REVENUE SCORE
# ============================================================

def calculate_revenue_score(
    intent,
    order_value,
    cart_value,
    previous_purchases,
    customer_tier
):
    """
    Calculate the commercial value of the current interaction.

    Formula:

    Revenue Score =
        0.40 * Order Value
      + 0.25 * Cart Value
      + 0.15 * Previous Purchases
      + 0.10 * Customer Tier
      + 0.10 * Intent Commercial Relevance

    This is a business score, NOT actual monetary revenue.
    """

    # --------------------------------------------------------
    # Customer signals
    # --------------------------------------------------------

    order_signal = normalize_order_value(
        order_value
    )

    cart_signal = normalize_cart_value(
        cart_value
    )

    purchase_history_signal = (
        normalize_previous_purchases(
            previous_purchases
        )
    )

    tier_signal = (
        normalize_customer_tier(
            customer_tier
        )
    )

    # --------------------------------------------------------
    # Intent commercial relevance
    #
    # Higher severity does NOT automatically mean
    # higher revenue.
    #
    # Here we use intent category knowledge to estimate
    # commercial relevance.
    # --------------------------------------------------------

    commercial_relevance = {

        "add_product": 0.90,

        "product_information": 0.85,

        "availability": 0.80,

        "availability_in_store": 0.75,

        "availability_online": 0.80,

        "sales_period": 0.90,

        "exchange_product": 0.65,

        "exchange_product_in_store": 0.60,

        "return_product": 0.45,

        "return_product_in_store": 0.45,

        "return_product_online": 0.45,

        "return_policy": 0.50,

        "track_order": 0.50,

        "track_delivery": 0.50,

        "delivery_time": 0.55,

        "delivery_issue": 0.60,

        "damaged_delivery": 0.50,

        "missing_item": 0.50,

        "wrong_item": 0.50,

        "shipping_costs": 0.60,

        "store_location": 0.45,

        "store_opening_hours": 0.30,

        "cancel_order": 0.30,

        "change_order": 0.55,

        "remove_product": 0.45,

        "order_history": 0.40,

        "request_invoice": 0.50,

        "pay": 0.90,

        "payment_methods": 0.75,

        "payment_issue": 0.80,

        "request_refund": 0.30,

        "refund_policy": 0.40,

        "refund_status": 0.35,

        "open_account": 0.65,

        "close_account": 0.20,

        "change_account": 0.40,

        "recover_password": 0.40,

        "request_right_to_rectification": 0.20,

        "customer_service": 0.50,

        "human_agent": 0.40,

        "submit_feedback": 0.35,

        "submit_product_feedback": 0.45,

        "submit_product_idea": 0.50,

        "technical_issue": 0.50,

        "product_issue": 0.55,
    }

    intent_revenue_signal = commercial_relevance.get(
        intent,
        0.50
    )

    # --------------------------------------------------------
    # Final formula
    # --------------------------------------------------------

    revenue_score = (

        0.40 * order_signal

        + 0.25 * cart_signal

        + 0.15 * purchase_history_signal

        + 0.10 * tier_signal

        + 0.10 * intent_revenue_signal
    )

    return round(
        clamp(revenue_score),
        4
    )


# ============================================================
# DYNAMIC RISK SCORE
# ============================================================

def calculate_risk_score(
    intent,
    order_value,
    confidence,
    normalized_entropy
):
    """
    Calculate risk dynamically.

    Formula:

    Risk Score =
        0.45 * Intent Severity
      + 0.25 * Order Value Exposure
      + 0.20 * Uncertainty
      + 0.10 * High-Value Exposure

    Higher score means greater potential business
    consequence if the interaction is mishandled.
    """

    # --------------------------------------------------------
    # Intent severity
    # --------------------------------------------------------

    intent_severity = get_intent_severity(
        intent
    )

    # --------------------------------------------------------
    # Order exposure
    # --------------------------------------------------------

    order_signal = normalize_order_value(
        order_value
    )

    # --------------------------------------------------------
    # Model uncertainty
    #
    # Confidence:
    #   1 = very confident
    #   0 = very uncertain
    #
    # Therefore:
    #
    # uncertainty = 1 - confidence
    # --------------------------------------------------------

    confidence = clamp(
        confidence
    )

    uncertainty = (
        1.0 - confidence
    )

    # --------------------------------------------------------
    # Entropy is already normalized 0-1.
    # --------------------------------------------------------

    normalized_entropy = clamp(
        normalized_entropy
    )

    # --------------------------------------------------------
    # Combined uncertainty signal
    # --------------------------------------------------------

    uncertainty_signal = (
        0.60 * uncertainty
        +
        0.40 * normalized_entropy
    )

    # --------------------------------------------------------
    # High-value exposure
    #
    # We emphasize large order values because a mistake
    # involving a high-value transaction can have greater
    # business consequences.
    # --------------------------------------------------------

    high_value_exposure = order_signal

    # --------------------------------------------------------
    # Final risk formula
    # --------------------------------------------------------

    risk_score = (

    0.50 * intent_severity

    + 0.25 * order_signal

    + 0.25 * uncertainty_signal
)
    return round(
        clamp(risk_score),
        4
    )


# ============================================================
# COMPLETE LAYER 2
# ============================================================

def calculate_business_values(
    intent,
    order_value,
    cart_value,
    previous_purchases,
    customer_tier,
    confidence,
    normalized_entropy
):
    """
    Calculate dynamic revenue and risk values
    for the current customer interaction.

    Returns:

    {
        "revenue_score": ...,
        "risk_score": ...,
        "intent_severity": ...,
        "uncertainty_signal": ...
    }
    """

    # --------------------------------------------------------
    # Uncertain intent
    # --------------------------------------------------------

    if intent == "uncertain_intent":

        # We don't know the actual business intent.
        #
        # We still calculate customer-side revenue value,
        # but use neutral intent assumptions.

        effective_intent = "customer_service"

    else:

        effective_intent = intent

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    revenue_score = calculate_revenue_score(

        intent=effective_intent,

        order_value=order_value,

        cart_value=cart_value,

        previous_purchases=previous_purchases,

        customer_tier=customer_tier
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_score = calculate_risk_score(

        intent=effective_intent,

        order_value=order_value,

        confidence=confidence,

        normalized_entropy=normalized_entropy
    )

    # --------------------------------------------------------
    # Additional diagnostic values
    # --------------------------------------------------------

    intent_severity = get_intent_severity(
        effective_intent
    )

    uncertainty_signal = (

        0.60 * (
            1.0 - clamp(confidence)
        )

        +

        0.40 * clamp(
            normalized_entropy
        )
    )

    return {

        "revenue_score":
            revenue_score,

        "risk_score":
            risk_score,

        "intent_severity":
            round(
                intent_severity,
                4
            ),

        "uncertainty_signal":
            round(
                uncertainty_signal,
                4
            )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "DYNAMIC LAYER 2: REVENUE + RISK TEST"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Example customer
    # --------------------------------------------------------

    customer = {

        "current_order_value": 5204.41,

        "cart_value": 5391.00,

        "previous_purchases": 5,

        "customer_tier": 1,
    }

    # --------------------------------------------------------
    # Example Layer-1 output
    # --------------------------------------------------------

    intent = "track_order"

    confidence = 0.3725

    normalized_entropy = 0.742

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    values = calculate_business_values(

        intent=intent,

        order_value=
            customer["current_order_value"],

        cart_value=
            customer["cart_value"],

        previous_purchases=
            customer["previous_purchases"],

        customer_tier=
            customer["customer_tier"],

        confidence=confidence,

        normalized_entropy=
            normalized_entropy
    )

    print(
        "\nIntent:",
        intent
    )

    print(
        f"Revenue Score : "
        f"{values['revenue_score']:.4f}"
    )

    print(
        f"Risk Score    : "
        f"{values['risk_score']:.4f}"
    )

    print(
        f"Intent Severity : "
        f"{values['intent_severity']:.4f}"
    )

    print(
        f"Uncertainty Signal : "
        f"{values['uncertainty_signal']:.4f}"
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "Layer 2 dynamic test completed."
    )

    print(
        "=" * 65
    )