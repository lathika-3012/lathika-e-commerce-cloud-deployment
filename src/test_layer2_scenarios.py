from revenue_risk_rules import calculate_business_values


# ============================================================
# LAYER 2 MULTI-SCENARIO VALIDATION
# ============================================================

SCENARIOS = [

    # --------------------------------------------------------
    # Scenario 1
    # Low-value customer + low-risk informational intent
    # --------------------------------------------------------
    {
        "name": "Low-value customer - Store opening hours",
        "intent": "store_opening_hours",
        "confidence": 0.90,
        "normalized_entropy": 0.10,
        "order_value": 200,
        "cart_value": 300,
        "previous_purchases": 1,
        "customer_tier": 0,
    },

    # --------------------------------------------------------
    # Scenario 2
    # High-value customer + tracking request
    # --------------------------------------------------------
    {
        "name": "High-value customer - Track order",
        "intent": "track_order",
        "confidence": 0.60,
        "normalized_entropy": 0.70,
        "order_value": 5204.41,
        "cart_value": 5391,
        "previous_purchases": 5,
        "customer_tier": 1,
    },

    # --------------------------------------------------------
    # Scenario 3
    # High-value customer + payment problem
    # --------------------------------------------------------
    {
        "name": "High-value customer - Payment issue",
        "intent": "payment_issue",
        "confidence": 0.55,
        "normalized_entropy": 0.75,
        "order_value": 5204.41,
        "cart_value": 5391,
        "previous_purchases": 5,
        "customer_tier": 1,
    },

    # --------------------------------------------------------
    # Scenario 4
    # High-value customer + refund request
    # --------------------------------------------------------
    {
        "name": "High-value customer - Request refund",
        "intent": "request_refund",
        "confidence": 0.55,
        "normalized_entropy": 0.75,
        "order_value": 5204.41,
        "cart_value": 5391,
        "previous_purchases": 5,
        "customer_tier": 1,
    },

    # --------------------------------------------------------
    # Scenario 5
    # High-value customer + product information
    # --------------------------------------------------------
    {
        "name": "High-value customer - Product information",
        "intent": "product_information",
        "confidence": 0.85,
        "normalized_entropy": 0.20,
        "order_value": 5204.41,
        "cart_value": 5391,
        "previous_purchases": 5,
        "customer_tier": 1,
    },

    # --------------------------------------------------------
    # Scenario 6
    # High-value customer + damaged delivery
    # --------------------------------------------------------
    {
        "name": "High-value customer - Damaged delivery",
        "intent": "damaged_delivery",
        "confidence": 0.55,
        "normalized_entropy": 0.75,
        "order_value": 5204.41,
        "cart_value": 5391,
        "previous_purchases": 5,
        "customer_tier": 1,
    },
]


# ============================================================
# RUN VALIDATION
# ============================================================

print("=" * 70)
print("LAYER 2 MULTI-SCENARIO VALIDATION")
print("=" * 70)


for scenario in SCENARIOS:

    # --------------------------------------------------------
    # Calculate Layer 2 values
    # --------------------------------------------------------

    business_values = calculate_business_values(

        intent=scenario["intent"],

        order_value=scenario["order_value"],

        cart_value=scenario["cart_value"],

        previous_purchases=scenario["previous_purchases"],

        customer_tier=scenario["customer_tier"],

        confidence=scenario["confidence"],

        normalized_entropy=scenario["normalized_entropy"],
    )


    # --------------------------------------------------------
    # Extract numeric values from dictionary
    # --------------------------------------------------------

    revenue_score = float(
        business_values["revenue_score"]
    )

    risk_score = float(
        business_values["risk_score"]
    )

    intent_severity = float(
        business_values["intent_severity"]
    )

    uncertainty_signal = float(
        business_values["uncertainty_signal"]
    )


    # --------------------------------------------------------
    # Display scenario
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        scenario["name"]
    )

    print("-" * 70)

    print(
        f"Intent              : "
        f"{scenario['intent']}"
    )

    print(
        f"Confidence          : "
        f"{scenario['confidence']:.2f}"
    )

    print(
        f"Order Value         : "
        f"₹{scenario['order_value']:.2f}"
    )

    print(
        f"Cart Value          : "
        f"₹{scenario['cart_value']:.2f}"
    )

    print(
        f"Previous Purchases  : "
        f"{scenario['previous_purchases']}"
    )

    print(
        f"Revenue Score       : "
        f"{revenue_score:.4f}"
    )

    print(
        f"Risk Score          : "
        f"{risk_score:.4f}"
    )

    print(
        f"Intent Severity     : "
        f"{intent_severity:.4f}"
    )

    print(
        f"Uncertainty Signal  : "
        f"{uncertainty_signal:.4f}"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)

print(
    "MULTI-SCENARIO VALIDATION COMPLETED"
)

print("=" * 70)