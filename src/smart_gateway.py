# ============================================================
# SMART AI GATEWAY
# E-COMMERCE CUSTOMER SUPPORT ROUTING
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from customer_database import (
    get_customer_profile
)

from revenue_risk_rules import (
    calculate_business_values
)

from profit_engine import (
    calculate_profit,
    calculate_routing_decision
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "distilbert_model"
)

PURCHASE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "purchase_model"
    / "purchase_probability_calibrated.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

INTENT_CONFIDENCE_THRESHOLD = 0.20

RELIABLE_INTENT_THRESHOLD = 0.60

INTENT_MARGIN_THRESHOLD = 0.05


# Internal demo customer.
# The customer does NOT enter this.
# In a real application this would come
# from the authenticated customer session.

DEMO_CUSTOMER_ID = "CUST0999"


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

print("Loading DistilBERT intent model...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH)
)

intent_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        str(MODEL_PATH)
    )
)

intent_model.eval()

print("Intent model loaded.")


# ============================================================
# LOAD PURCHASE MODEL
# ============================================================

print("Loading purchase probability model...")

purchase_model = joblib.load(
    PURCHASE_MODEL_PATH
)

print("Purchase model loaded.")


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

intent_model.to(DEVICE)


# ============================================================
# LAYER 1
# INTENT DETECTION
# ============================================================

def predict_intent(query):

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = intent_model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]


    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=3
        )
    )

    id2label = (
        intent_model
        .config
        .id2label
    )

    top_predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        label_id = index.item()

        confidence = (
            probability.item()
        )

        # Handles both integer and string keys
        if label_id in id2label:

            intent = id2label[
                label_id
            ]

        else:

            intent = id2label[
                str(label_id)
            ]

        top_predictions.append(
            {
                "intent": intent,
                "confidence": confidence
            }
        )


    # --------------------------------------------------------
    # BEST INTENT
    # --------------------------------------------------------

    best_intent = (
        top_predictions[0]["intent"]
    )

    best_confidence = (
        top_predictions[0]["confidence"]
    )


    # --------------------------------------------------------
    # TOP-1 / TOP-2 MARGIN
    # --------------------------------------------------------

    intent_margin = (
        top_predictions[0]["confidence"]
        -
        top_predictions[1]["confidence"]
    )


    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    entropy = -torch.sum(
        probabilities
        *
        torch.log(
            probabilities + 1e-12
        )
    ).item()

    num_classes = (
        probabilities.shape[0]
    )

    max_entropy = np.log(
        num_classes
    )

    normalized_entropy = (
        entropy / max_entropy
        if max_entropy > 0
        else 0.0
    )


    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    is_uncertain = (
        best_confidence
        <
        INTENT_CONFIDENCE_THRESHOLD
    )


    # --------------------------------------------------------
    # RELIABLE INTENT
    # --------------------------------------------------------

    is_reliable = (
        best_confidence
        >=
        RELIABLE_INTENT_THRESHOLD
    )


    # --------------------------------------------------------
    # NON-ACTIONABLE QUERY
    # --------------------------------------------------------

    out_of_scope = (
        best_confidence < 0.20
        and
        intent_margin < 0.05
        and
        normalized_entropy >= 0.85
    )

    non_actionable = out_of_scope


    # --------------------------------------------------------
    # FINAL INTENT
    # --------------------------------------------------------

    if is_uncertain:

        final_intent = (
            "uncertain_intent"
        )

    else:

        final_intent = best_intent


    return {

        "intent":
            final_intent,

        "raw_intent":
            best_intent,

        "confidence":
            best_confidence,

        "is_uncertain":
            is_uncertain,

        "is_reliable":
            is_reliable,

        "intent_margin":
            intent_margin,

        "entropy":
            entropy,

        "normalized_entropy":
            normalized_entropy,

        "non_actionable":
            non_actionable,

        "top_predictions":
            top_predictions
    }


# ============================================================
# PRINT LAYER 1 RESULTS
# ============================================================

def print_intent_results(
    intent_result
):

    print(
        "\n--- LAYER 1: INTENT DETECTION ---"
    )

    print(
        f"Final Intent : "
        f"{intent_result['intent']}"
    )

    print(
        f"Raw Intent   : "
        f"{intent_result['raw_intent']}"
    )

    print(
        f"Confidence   : "
        f"{intent_result['confidence'] * 100:.2f}%"
    )

    print(
        f"Intent Margin: "
        f"{intent_result['intent_margin'] * 100:.2f}%"
    )

    print(
        f"Entropy      : "
        f"{intent_result['entropy']:.4f}"
    )

    print(
        f"Normalized Entropy : "
        f"{intent_result['normalized_entropy']:.4f}"
    )

    print(
        "\nTop 3 Predictions:"
    )

    for index, prediction in enumerate(
        intent_result["top_predictions"],
        start=1
    ):

        print(
            f"{index}. "
            f"{prediction['intent']} "
            f"→ "
            f"{prediction['confidence'] * 100:.2f}%"
        )


    if intent_result["non_actionable"]:

        print(
            "\nQuery Status : "
            "NON-ACTIONABLE / "
            "NO RELIABLE E-COMMERCE INTENT"
        )

    elif intent_result["is_uncertain"]:

        print(
            "\nWARNING: "
            "Model confidence is very low."
        )

        print(
            "Query is treated as UNCERTAIN."
        )

    else:

        print(
            "\nQuery Status : "
            "E-COMMERCE INTENT DETECTED"
        )


# ============================================================
# LAYER 3
# PURCHASE PROBABILITY
# ============================================================

def predict_purchase_probability(
    customer
):

    features = pd.DataFrame(
        [{
            "product_views":
                customer["product_views"],

            "cart_additions":
                customer["cart_additions"],

            "previous_purchases":
                customer["previous_purchases"],

            "previous_sessions":
                customer["previous_sessions"],

            "favorites":
                customer["favorites"],

            "time_spent_minutes":
                customer["time_spent_minutes"],

            "cart_value":
                customer["cart_value"],

            "checkout_attempts":
                customer["checkout_attempts"],

            "customer_tier":
                customer["customer_tier"],
        }]
    )


    # --------------------------------------------------------
    # CASE 1
    # Pipeline / calibrated model
    # --------------------------------------------------------

    if hasattr(
        purchase_model,
        "predict_proba"
    ):

        probability = (
            purchase_model
            .predict_proba(
                features
            )[0][1]
        )

        return float(
            probability
        )


    # --------------------------------------------------------
    # CASE 2
    # Dictionary containing model + scaler
    # --------------------------------------------------------

    if isinstance(
        purchase_model,
        dict
    ):

        model = (
            purchase_model
            .get("model")
        )

        scaler = (
            purchase_model
            .get("scaler")
        )

        if model is None:

            raise ValueError(
                "Purchase model dictionary "
                "does not contain 'model'."
            )

        if scaler is not None:

            scaled_features = (
                scaler.transform(
                    features
                )
            )

        else:

            scaled_features = features

        probability = (
            model
            .predict_proba(
                scaled_features
            )[0][1]
        )

        return float(
            probability
        )


    raise ValueError(
        "Unsupported purchase model format."
    )


# ============================================================
# CUSTOMER TARGETING
# ============================================================

def get_customer_category(
    purchase_probability
):

    if purchase_probability >= 0.80:

        return "HOT"

    elif purchase_probability >= 0.40:

        return "WARM"

    else:

        return "COLD"


# ============================================================
# MAIN SMART GATEWAY
# ============================================================

def run_gateway():

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "SMART AI GATEWAY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nInternal Demo Customer: "
        f"{DEMO_CUSTOMER_ID}"
    )

    print(
        "Customer only needs to enter the query."
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        print("\n")


        # ====================================================
        # CUSTOMER QUERY
        # ====================================================

        query = input(
            "Customer Query: "
        ).strip()

        if query.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nSmart AI Gateway stopped."
            )

            break


        if not query:

            print(
                "Please enter a customer query."
            )

            continue


        # ====================================================
        # CUSTOMER DATABASE
        # ====================================================

        customer = (
            get_customer_profile(
                DEMO_CUSTOMER_ID
            )
        )

        if customer is None:

            print(
                f"\nERROR: Customer "
                f"{DEMO_CUSTOMER_ID} "
                f"was not found in the database."
            )

            break


        # ====================================================
        # CURRENT ORDER VALUE
        # ====================================================

        order_value = customer.get(
            "current_order_value"
        )

        if order_value is None:

            print(
                "\nERROR: current_order_value "
                "is missing from the customer database."
            )

            print(
                "Run add_order_value.py first."
            )

            break

        order_value = float(
            order_value
        )


        # ====================================================
        # LAYER 1
        # ====================================================

        intent_result = (
            predict_intent(
                query
            )
        )

        print_intent_results(
            intent_result
        )

        intent = (
            intent_result["intent"]
        )

        confidence = (
            intent_result["confidence"]
        )

        intent_margin = (
            intent_result["intent_margin"]
        )

        normalized_entropy = (
            intent_result[
                "normalized_entropy"
            ]
        )

        is_uncertain = (
            intent_result[
                "is_uncertain"
            ]
        )

        non_actionable = (
            intent_result[
                "non_actionable"
            ]
        )


        # ====================================================
        # LAYER 2
        # DYNAMIC REVENUE + RISK
        # ====================================================

        # IMPORTANT:
        # We use the dynamic formula-based Layer 2.
        #
        # Revenue and risk are calculated using:
        # - current order value
        # - cart value
        # - previous purchases
        # - customer tier
        # - predicted intent
        # - confidence
        # - normalized entropy
        #
        # Purchase probability is NOT passed here.
        # That belongs to Layer 3.


        business_values = (
            calculate_business_values(

                intent=intent,

                order_value=order_value,

                cart_value=float(
                    customer.get(
                        "cart_value",
                        0
                    )
                ),

                previous_purchases=int(
                    customer.get(
                        "previous_purchases",
                        0
                    )
                ),

                customer_tier=int(
                    customer.get(
                        "customer_tier",
                        0
                    )
                ),

                confidence=confidence,

                normalized_entropy=
                    normalized_entropy
            )
        )


        revenue_score = float(
            business_values[
                "revenue_score"
            ]
        )

        risk_score = float(
            business_values[
                "risk_score"
            ]
        )

        intent_severity = float(
            business_values[
                "intent_severity"
            ]
        )

        uncertainty_signal = float(
            business_values[
                "uncertainty_signal"
            ]
        )


        print(
            "\n--- LAYER 2: "
            "DYNAMIC REVENUE + RISK ---"
        )

        print(
            f"Revenue Score      : "
            f"{revenue_score:.4f}"
        )

        print(
            f"Risk Score         : "
            f"{risk_score:.4f}"
        )

        print(
            f"Intent Severity    : "
            f"{intent_severity:.4f}"
        )

        print(
            f"Uncertainty Signal : "
            f"{uncertainty_signal:.4f}"
        )


        # ====================================================
        # LAYER 3
        # PURCHASE PROBABILITY
        # ====================================================

        purchase_probability = (
            predict_purchase_probability(
                customer
            )
        )

        customer_category = (
            get_customer_category(
                purchase_probability
            )
        )


        print(
            "\n--- LAYER 3: "
            "PURCHASE PROBABILITY ---"
        )

        print(
            f"Purchase Probability : "
            f"{purchase_probability * 100:.2f}%"
        )

        print(
            f"Customer Category    : "
            f"{customer_category}"
        )


        # ====================================================
        # PROFIT ENGINE
        # ====================================================

        # THIS IS THE IMPORTANT FIX.
        #
        # calculate_profit() requires:
        #
        # 1. purchase_probability
        # 2. order_value
        # 3. risk_score
        # 4. route


        small_profit = (
            calculate_profit(
                purchase_probability,
                order_value,
                risk_score,
                "small_model"
            )
        )

        llm_profit = (
            calculate_profit(
                purchase_probability,
                order_value,
                risk_score,
                "llm"
            )
        )

        human_profit = (
            calculate_profit(
                purchase_probability,
                order_value,
                risk_score,
                "llm_human"
            )
        )


        print(
            "\n--- PROFIT ENGINE ---"
        )

        print(
            f"Current Order Value : "
            f"₹{order_value:.2f}"
        )

        print(
            f"small_model     Profit = "
            f"₹{small_profit:.2f}"
        )

        print(
            f"llm             Profit = "
            f"₹{llm_profit:.2f}"
        )

        print(
            f"llm_human       Profit = "
            f"₹{human_profit:.2f}"
        )


        # ====================================================
        # FINAL INTELLIGENT ROUTER
        # ====================================================

        routing = (
            calculate_routing_decision(

                confidence=
                    confidence,

                revenue_weight=
                    revenue_score,

                risk_score=
                    risk_score,

                purchase_probability=
                    purchase_probability,

                small_profit=
                    small_profit,

                llm_profit=
                    llm_profit,

                human_profit=
                    human_profit,

                is_uncertain=
                    is_uncertain,

                intent_margin=
                    intent_margin,

                normalized_entropy=
                    normalized_entropy,

                non_actionable=
                    non_actionable, 
                query=
                    query
            )
        )

        # ====================================================
        # INCREMENTAL VALUE
        # ====================================================

        print(
            "\n--- INCREMENTAL VALUE ---"
        )

        print(
            f"LLM Gain over Small Model : "
            f"₹{routing['llm_gain']:.2f}"
        )

        print(
            f"Human Gain over LLM       : "
            f"₹{routing['human_gain']:.2f}"
        )


        # ====================================================
        # FINAL ROUTER
        # ====================================================

        print(
            "\n--- FINAL ROUTER ---"
        )

        print(
            f"Selected Route : "
            f"{routing['route']}"
        )

        print(
            f"Reason         : "
            f"{routing['reason']}"
        )


        # ====================================================
        # FINAL CUSTOMER TARGETING
        # ====================================================

        print(
            "\n--- FINAL CUSTOMER TARGETING ---"
        )

        print(
            f"Customer Category : "
            f"{customer_category}"
        )

        print(
            "\n"
            +
            "=" * 70
        )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    run_gateway()