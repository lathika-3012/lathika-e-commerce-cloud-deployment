# ============================================================
# PROFIT ENGINE
# E-COMMERCE SMART AI GATEWAY
# ============================================================

import re


# ============================================================
# ROUTE CONFIGURATION
# ============================================================

ROUTE_CONFIG = {

    "small_model": {
        "cost": 0.01,
        "correctness": 0.95,
    },

    "llm": {
        "cost": 0.10,
        "correctness": 0.99,
    },

    "llm_human": {
        "cost": 0.50,
        "correctness": 0.995,
    },
}


# ============================================================
# PROFIT CALCULATION
# ============================================================

def calculate_profit(
    purchase_probability,
    order_value,
    risk_score,
    route,
    risk_penalty_rate=0.20
):
    """
    Calculate expected profit for a routing option.

    This function only calculates profit.
    It does NOT decide the final route.
    """

    if route not in ROUTE_CONFIG:

        raise ValueError(
            f"Unknown route: {route}. "
            f"Expected one of: {list(ROUTE_CONFIG.keys())}"
        )

    config = ROUTE_CONFIG[route]

    # --------------------------------------------------------
    # Expected revenue
    # --------------------------------------------------------

    expected_revenue = (
        purchase_probability
        *
        order_value
        *
        config["correctness"]
    )

    # --------------------------------------------------------
    # Risk penalty
    # --------------------------------------------------------

    risk_penalty = (
        risk_score
        *
        order_value
        *
        risk_penalty_rate
    )

    # --------------------------------------------------------
    # Expected profit
    # --------------------------------------------------------

    profit = (
        expected_revenue
        -
        config["cost"]
        -
        risk_penalty
    )

    return round(
        profit,
        2
    )


# ============================================================
# SENSITIVE QUERY DETECTOR
# ============================================================

def detect_sensitive_query(query):
    """
    Detect potentially sensitive or high-impact queries.

    Categories:

    1. Legal / court
    2. Fraud / scam
    3. Payment disputes
    4. Unauthorized transactions
    5. Account security
    6. Threats / authorities
    7. Serious complaints
    8. Privacy / data
    9. Serious product issues
    """

    query = query.lower().strip()

    # ========================================================
    # LEGAL / COURT
    # ========================================================

    legal_patterns = [

        r"\blegal\b",
        r"\blaw\b",
        r"\blawyer\b",
        r"\battorney\b",
        r"\bcourt\b",
        r"\blawsuit\b",
        r"\blitigation\b",

        r"\bsue\b",
        r"\bsued\b",
        r"\bsuing\b",

        r"\blegal action\b",
        r"\blegal case\b",
        r"\blegal complaint\b",

        r"\bfile a case\b",
        r"\bfile case\b",

        r"\bfile a lawsuit\b",
        r"\bfile lawsuit\b",

        r"\btake legal action\b",
        r"\btake you to court\b",
        r"\bgo to court\b",

        r"\bconsumer court\b",
        r"\bconsumer forum\b",

        r"\blegal notice\b",
        r"\bsend a legal notice\b",
        r"\bserve legal notice\b",

        r"\bconsumer protection\b",
        r"\bconsumer rights\b",
    ]

    # ========================================================
    # FRAUD / SCAM
    # ========================================================

    fraud_patterns = [

        r"\bfraud\b",
        r"\bfraudulent\b",

        r"\bscam\b",
        r"\bscammed\b",
        r"\bscamming\b",

        r"\bcheated\b",
        r"\bcheating\b",

        r"\bdeceived\b",
        r"\bdeception\b",

        r"\bfake\b",
        r"\bforged\b",
        r"\bforgery\b",

        r"\bfraudulent transaction\b",
        r"\bfraudulent payment\b",
        r"\bfraudulent charge\b",
        r"\bfraudulent order\b",
        r"\bfraudulent activity\b",

        r"\bscam company\b",
        r"\bscam website\b",

        r"\byou scammed me\b",
        r"\byou cheated me\b",
    ]

    # ========================================================
    # PAYMENT DISPUTES
    # ========================================================

    payment_patterns = [

        r"\bcharged twice\b",
        r"\bcharged double\b",
        r"\bdouble charged\b",
        r"\bcharged two times\b",
        r"\bcharged multiple times\b",
        r"\bcharged more than once\b",

        r"\bduplicate charge\b",
        r"\bduplicate payment\b",
        r"\bduplicate transaction\b",

        r"\bwrong charge\b",
        r"\bwrong payment\b",

        r"\bincorrect charge\b",
        r"\bincorrect payment\b",

        r"\bextra charge\b",
        r"\bextra payment\b",

        r"\bunknown charge\b",
        r"\bunknown payment\b",
        r"\bunknown transaction\b",

        r"\bpayment dispute\b",
        r"\bdispute payment\b",
        r"\bdispute charge\b",
        r"\bcharge dispute\b",

        r"\bchargeback\b",
        r"\bcharge back\b",

        r"\bpayment reversed\b",
        r"\bpayment reversal\b",
    ]

    # ========================================================
    # UNAUTHORIZED TRANSACTIONS / SECURITY
    # ========================================================

    security_patterns = [

        r"\bunauthorized\b",
        r"\bunauthorised\b",

        r"\bnot my transaction\b",
        r"\bnot my payment\b",
        r"\bnot my purchase\b",

        r"\bdid not make this purchase\b",
        r"\bdidn't make this purchase\b",

        r"\bdid not authorize\b",
        r"\bdidn't authorize\b",

        r"\bdid not authorise\b",
        r"\bdidn't authorise\b",

        r"\baccount hacked\b",
        r"\bhacked account\b",

        r"\bmy account was hacked\b",
        r"\baccount compromised\b",
        r"\baccount stolen\b",

        r"\bidentity theft\b",
        r"\bidentity stolen\b",

        r"\bpassword stolen\b",

        r"\bcredit card stolen\b",
        r"\bcard stolen\b",
        r"\bcard compromised\b",

        r"\bsomeone used my card\b",
        r"\bsomeone used my account\b",

        r"\bsomeone made a purchase\b",

        r"\bunknown purchase\b",
        r"\bunknown order\b",

        r"\bcard used without permission\b",
        r"\bpayment without permission\b",
        r"\bpurchase without permission\b",
    ]

    # ========================================================
    # THREATS / AUTHORITIES
    # ========================================================

    threat_patterns = [

        r"\bthreat\b",
        r"\bthreaten\b",
        r"\bthreatening\b",

        r"\bpolice\b",
        r"\bpolice complaint\b",
        r"\bpolice case\b",

        r"\breport you\b",
        r"\breport your company\b",
        r"\breport this company\b",

        r"\breport to authorities\b",

        r"\bauthorities\b",

        r"\bregulator\b",
        r"\bregulatory complaint\b",

        r"\bgovernment complaint\b",
    ]

    # ========================================================
    # SERIOUS COMPLAINTS
    # ========================================================

    complaint_patterns = [

        r"\bformal complaint\b",
        r"\bserious complaint\b",
        r"\bofficial complaint\b",

        r"\bescalate this\b",
        r"\bescalate my complaint\b",
        r"\bescalate this issue\b",

        r"\bfile a complaint\b",
        r"\bfile complaint\b",

        r"\bcomplaint against\b",
        r"\bcomplaint about your company\b",
        r"\bcomplaint against your company\b",

        r"\bcomplaint to management\b",

        r"\bspeak to management\b",
        r"\bspeak to a manager\b",

        r"\bmanager complaint\b",
        r"\bsupervisor complaint\b",
    ]

    # ========================================================
    # PRIVACY / DATA
    # ========================================================

    privacy_patterns = [

        r"\bprivacy violation\b",
        r"\bprivacy issue\b",
        r"\bprivacy complaint\b",

        r"\bdata breach\b",
        r"\bdata leak\b",

        r"\bmy data was leaked\b",

        r"\bpersonal data\b",
        r"\bpersonal information\b",
        r"\bprivate information\b",

        r"\bdata protection\b",
        r"\bdata misuse\b",

        r"\bmisuse my data\b",

        r"\bdelete my data\b",
        r"\bremove my data\b",

        r"\bdata stolen\b",
        r"\bmy information was stolen\b",
    ]

    # ========================================================
    # SERIOUS PRODUCT / SAFETY ISSUES
    # ========================================================

    serious_product_patterns = [

        r"\bwrong product\b",
        r"\bfake product\b",
        r"\bcounterfeit\b",

        r"\bdamaged product\b",

        r"\bdangerous product\b",
        r"\bunsafe product\b",

        r"\binjured\b",
        r"\binjury\b",
        r"\bharmed\b",
        r"\bharm caused\b",

        r"\bproduct caused injury\b",
        r"\bproduct hurt me\b",
    ]

    # ========================================================
    # COMBINE ALL PATTERNS
    # ========================================================

    all_patterns = (
        legal_patterns
        + fraud_patterns
        + payment_patterns
        + security_patterns
        + threat_patterns
        + complaint_patterns
        + privacy_patterns
        + serious_product_patterns
    )

    # ========================================================
    # FIND MATCHES
    # ========================================================

    matched_patterns = []

    for pattern in all_patterns:

        if re.search(
            pattern,
            query
        ):

            matched_patterns.append(
                pattern
            )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "is_sensitive":
            len(matched_patterns) > 0,

        "matched_count":
            len(matched_patterns)
    }


# ============================================================
# FINAL INTELLIGENT ROUTING
# ============================================================

def calculate_routing_decision(
    confidence,
    revenue_weight,
    risk_score,
    purchase_probability,
    small_profit,
    llm_profit,
    human_profit,
    is_uncertain,
    intent_margin,
    normalized_entropy,
    non_actionable,
    query=""
):
    """
    Final routing decision.

    Possible routes:

        small_model
        llm
        llm_human

    Routing considers:

        - Intent confidence
        - Intent margin
        - Entropy
        - Revenue value
        - Risk
        - Purchase probability
        - Expected profit
        - Sensitive query detection
    """

    # ========================================================
    # INCREMENTAL VALUES
    # ========================================================

    llm_gain = (
        llm_profit
        -
        small_profit
    )

    human_gain = (
        human_profit
        -
        llm_profit
    )

    # ========================================================
    # SENSITIVE QUERY DETECTION
    # ========================================================

    sensitive_result = detect_sensitive_query(
        query
    )

    sensitive_query = (
        sensitive_result["is_sensitive"]
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    HIGH_RISK = 0.70

    CRITICAL_RISK = 0.85

    HIGH_REVENUE = 0.70

    VERY_LOW_CONFIDENCE = 0.20

    AMBIGUOUS_MARGIN = 0.15

    HIGH_ENTROPY = 0.85

    # ========================================================
    # ROUTING SIGNALS
    # ========================================================

    print(
        "\n--- ROUTING SIGNALS ---"
    )

    print(
        f"Confidence           : "
        f"{confidence:.4f}"
    )

    print(
        f"Intent Margin        : "
        f"{intent_margin:.4f}"
    )

    print(
        f"Normalized Entropy   : "
        f"{normalized_entropy:.4f}"
    )

    print(
        f"Revenue Weight       : "
        f"{revenue_weight:.4f}"
    )

    print(
        f"Risk Score           : "
        f"{risk_score:.4f}"
    )

    print(
        f"Purchase Probability : "
        f"{purchase_probability:.4f}"
    )

    print(
        f"LLM Gain             : "
        f"₹{llm_gain:.2f}"
    )

    print(
        f"Human Gain           : "
        f"₹{human_gain:.2f}"
    )

    print(
        f"Sensitive Query      : "
        f"{sensitive_query}"
    )

    # ========================================================
    # CASE 1
    # CRITICAL RISK -> LLM + HUMAN
    # ========================================================

    if (
        risk_score >= CRITICAL_RISK
        and
        purchase_probability >= 0.60
        and
        human_gain > 0
    ):

        return {

            "route":
                "llm_human",

            "reason":
                "Critical risk detected. "
                "Human escalation provides "
                "additional expected value.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 2
    # SENSITIVE QUERY -> LLM
    # ========================================================

    if (
        sensitive_query
        and
        llm_gain > 0
    ):

        return {

            "route":
                "llm",

            "reason":
                "Sensitive or potentially "
                "high-impact query detected. "
                "LLM is selected for deeper "
                "and safer handling.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 3
    # HIGH RISK -> LLM
    # ========================================================

    if (
        risk_score >= HIGH_RISK
        and
        llm_gain > 0
    ):

        return {

            "route":
                "llm",

            "reason":
                "High-risk query detected. "
                "LLM provides higher expected "
                "value.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 4
    # UNCERTAIN BUT NOT SENSITIVE
    #
    # IMPORTANT:
    # We intentionally do NOT route uncertain queries
    # to LLM simply because the customer is valuable.
    #
    # Example:
    #
    # "good quality"
    #
    # -> small_model
    # ========================================================

    if (
        is_uncertain
        and
        intent_margin < AMBIGUOUS_MARGIN
    ):

        return {

            "route":
                "small_model",

            "reason":
                "The query is ambiguous but "
                "no sensitive issue was detected. "
                "Small model is sufficient.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 5
    # NON-ACTIONABLE NORMAL QUERY -> SMALL MODEL
    # ========================================================

    if non_actionable:

        return {

            "route":
                "small_model",

            "reason":
                "No reliable e-commerce intent "
                "was detected and no sensitive "
                "issue was identified. Small "
                "model is sufficient.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 6
    # HIGH BUSINESS VALUE
    #
    # This applies only when the query has a recognizable
    # e-commerce intent.
    # ========================================================

    if (
        revenue_weight >= HIGH_REVENUE
        and
        llm_gain > 0
    ):

        return {

            "route":
                "llm",

            "reason":
                "High business value detected "
                "and LLM provides higher "
                "expected profit.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 7
    # PROFIT-BASED ROUTING
    # ========================================================

    if (
        llm_profit
        >
        small_profit
    ):

        return {

            "route":
                "llm",

            "reason":
                "LLM provides higher expected "
                "profit than the small model.",

            "llm_gain":
                round(
                    llm_gain,
                    2
                ),

            "human_gain":
                round(
                    human_gain,
                    2
                ),
        }

    # ========================================================
    # CASE 8
    # DEFAULT -> SMALL MODEL
    # ========================================================

    return {

        "route":
            "small_model",

        "reason":
            "Small model provides sufficient "
            "capability for this query.",

        "llm_gain":
            round(
                llm_gain,
                2
            ),

        "human_gain":
            round(
                human_gain,
                2
            ),
    }


# ============================================================
# PROFIT ENGINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "PROFIT ENGINE TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Test values
    # --------------------------------------------------------

    purchase_probability = 0.85

    order_value = 2000

    risk_score = 0.30

    # --------------------------------------------------------
    # Calculate profits
    # --------------------------------------------------------

    small_profit = calculate_profit(
        purchase_probability,
        order_value,
        risk_score,
        "small_model"
    )

    llm_profit = calculate_profit(
        purchase_probability,
        order_value,
        risk_score,
        "llm"
    )

    human_profit = calculate_profit(
        purchase_probability,
        order_value,
        risk_score,
        "llm_human"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"\nSmall Model Profit : "
        f"₹{small_profit}"
    )

    print(
        f"LLM Profit         : "
        f"₹{llm_profit}"
    )

    print(
        f"LLM + Human Profit : "
        f"₹{human_profit}"
    )

    print(
        f"\nLLM Gain           : "
        f"₹{llm_profit - small_profit:.2f}"
    )

    print(
        f"Human Gain         : "
        f"₹{human_profit - llm_profit:.2f}"
    )

    print(
        "=" * 70
    )