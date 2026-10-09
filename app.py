"""Streamlit dashboard for the Profit & Risk-Aware E-Commerce AI Gateway.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from config import BENCHMARK_CSV_PATH, BENCHMARK_JSON_PATH, get_settings
from schemas import ChatRequest, ChatResponse, CustomerProfile

settings = get_settings()
st.set_page_config(page_title="E-Commerce AI Gateway", page_icon="🛒", layout="wide")

MODE_AUTO = "Auto (API if running, else local)"
MODE_API = "FastAPI backend only"
MODE_LOCAL = "Local pipeline only"


@st.cache_resource(show_spinner="Loading gateway pipeline...")
def get_local_pipeline():
    """Create the pipeline once per Streamlit server process."""
    from pipeline import GatewayPipeline

    return GatewayPipeline()


def call_gateway(mode: str, api_url: str, request: ChatRequest) -> Tuple[ChatResponse, str, Dict[str, Any]]:
    """Return ``(response, engine_used, telemetry_summary)``."""
    if mode in (MODE_AUTO, MODE_API):
        base = api_url.rstrip("/")
        try:
            resp = requests.post(f"{base}/api/v1/chat", json=request.model_dump(), timeout=60)
            if resp.status_code == 422:
                raise ValueError("The API rejected the input: " + json.dumps(resp.json().get("details", [])))
            resp.raise_for_status()
            summary = requests.get(f"{base}/api/v1/telemetry/summary", timeout=10).json()
            return ChatResponse(**resp.json()), f"FastAPI ({base})", summary
        except requests.RequestException:
            if mode == MODE_API:
                raise
    pipeline = get_local_pipeline()
    result = pipeline.handle(request)
    return result, "Local pipeline", pipeline.telemetry.summary()


def load_benchmark() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Read the benchmark CSV (and JSON metadata) if the benchmark has been run."""
    if not BENCHMARK_CSV_PATH.exists():
        return None, None
    df = pd.read_csv(BENCHMARK_CSV_PATH)
    meta = None
    if BENCHMARK_JSON_PATH.exists():
        try:
            meta = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
    return df, meta


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Customer Input")
    query = st.text_area("Customer Query", value="I want a laptop for gaming", height=90)
    cart_value = st.number_input("Cart Value ($)", min_value=0.0, value=1200.0, step=50.0)
    tier = st.selectbox("Customer Tier", ["VIP", "GOLD", "SILVER", "BRONZE"], index=1)
    page_views = st.number_input("Recent Page Views (last 10 min)", min_value=0, value=8, step=1)
    analyze = st.button("Analyze & Route", type="primary")
    st.divider()
    engine_mode = st.radio("Engine", [MODE_AUTO, MODE_API, MODE_LOCAL])
    api_url = st.text_input("API URL", value=settings.API_URL)

st.title("🛒 Profit & Risk-Aware E-Commerce AI Gateway")
st.caption("Routes each customer query to a cheap SLM or an expensive LLM using intent risk, "
           "purchase probability and expected profit. Costs are simulated demonstration values.")

tab_route, tab_bench, tab_arch = st.tabs(["Routing Console", "6-Model Benchmark", "Architecture"])

# --------------------------------------------------------------------------- #
# Routing console
# --------------------------------------------------------------------------- #
with tab_route:
    if "history" not in st.session_state:
        st.session_state.history = []

    if analyze:
        try:
            request = ChatRequest(query=query, profile=CustomerProfile(
                cart_value=cart_value, tier=tier, page_views_last_10m=int(page_views)))
            result, engine, summary = call_gateway(engine_mode, api_url, request)
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
        except requests.RequestException:
            st.error(f"Could not reach the API at {api_url}. Start it with "
                     "`python -m uvicorn main:app --reload` or choose 'Local pipeline only'.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {type(exc).__name__}")
        else:
            st.session_state.last = (result, engine, summary)
            st.session_state.history.append({
                "query": query[:60], "intent": result.intent, "segment": result.segment,
                "model": result.selected_model, "expected_profit": result.expected_profit,
                "latency_ms": result.latency})

    if "last" not in st.session_state:
        st.info("Enter a query in the sidebar and press **Analyze & Route**.")
    else:
        result, engine, summary = st.session_state.last
        st.caption(f"Engine: {engine}  |  Intent engine: {result.classifier_backend}")
        if result.classifier_backend != "transformer":
            st.warning("No fine-tuned model found - using the keyword fallback. "
                       "Run `python benchmark_6_models.py --quick` to train the real intent model.")

        st.subheader("Intent")
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected Intent", result.intent)
        c2.metric("Confidence", f"{result.confidence:.1%}")
        c3.metric("Canonical Intent / Revenue", f"{result.canonical_intent} / {result.revenue_weight}")

        st.subheader("Customer")
        c1, c2, c3 = st.columns(3)
        c1.metric("Customer Segment", result.segment)
        c2.metric("Purchase Probability", f"{result.purchase_probability:.1%}")
        c3.metric("Risk Score", f"{result.risk_score:.2f}")

        st.subheader("Financial")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Profit (if escalated)", f"${result.expected_profit:,.2f}")
        c2.metric("Model Cost (simulated)", f"${result.model_cost:.3f}")
        c3.metric("Estimated Savings (this query)", f"${result.estimated_savings:.3f}")

        st.subheader("Routing")
        c1, c2 = st.columns([1, 3])
        c1.metric("Selected Model", result.selected_model)
        (st.success if result.selected_model == "SLM" else st.warning)(result.reason)

        st.subheader("Performance")
        st.metric("Latency", f"{result.latency:.2f} ms")

        st.subheader("Response")
        st.write(result.response)
        st.caption(f"Response source: {result.response_source}")

        st.subheader("Cumulative telemetry")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", summary.get("total_requests", 0))
        c2.metric("LLM share", f"{summary.get('llm_share_percent', 0)}%")
        c3.metric("Actual cost (sim.)", f"${summary.get('actual_cost', 0):.4f}")
        c4.metric("Estimated/Theoretical Cost Savings", f"${summary.get('estimated_cost_savings', 0):.4f}")
        st.caption("Savings compare simulated costs with a baseline that sends 100% of traffic to the LLM; "
                   "they are not real-world financial savings.")

        with st.expander("Session history"):
            st.dataframe(pd.DataFrame(st.session_state.history))

# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
with tab_bench:
    bench_df, bench_meta = load_benchmark()
    if bench_df is None:
        st.info("No benchmark results yet. Run `python benchmark_6_models.py --quick` first.")
    else:
        production = bench_df.loc[bench_df["selected_production"] == True, "model"]  # noqa: E712
        winner = bench_df.loc[bench_df["overall_winner"] == True, "model"]  # noqa: E712
        if len(production):
            st.success(f"Selected production model: **{production.iloc[0]}**")
        if len(winner):
            st.caption(f"Highest overall selection score: {winner.iloc[0]}  |  "
                       f"Mode: {bench_df['mode'].iloc[0]}  |  Device: {bench_df['device'].iloc[0]}")
        if bench_meta and bench_meta.get("selection", {}).get("production_rationale"):
            st.caption(bench_meta["selection"]["production_rationale"])

        shown = bench_df[["model", "accuracy", "weighted_f1", "precision", "recall", "latency_ms",
                          "size_mb", "train_time_s", "parameters", "selection_score", "selected_production"]]
        try:
            st.dataframe(shown.style.apply(
                lambda row: ["background-color: #d4edda" if row["selected_production"] else "" for _ in row], axis=1))
        except Exception:  # noqa: BLE001 - styling is optional
            st.dataframe(shown)

        def highlight_chart(column: str, title: str) -> None:
            st.markdown(f"**{title}**")
            is_prod = bench_df["selected_production"] == True  # noqa: E712
            chart = pd.DataFrame({
                "Selected production model": bench_df[column].where(is_prod, 0.0).values,
                "Other models": bench_df[column].where(~is_prod, 0.0).values}, index=bench_df["model"])
            st.bar_chart(chart)

        left, right = st.columns(2)
        with left:
            highlight_chart("weighted_f1", "Weighted F1 (higher is better)")
            highlight_chart("latency_ms", "Latency, ms per query (lower is better)")
        with right:
            highlight_chart("accuracy", "Accuracy (higher is better)")
            highlight_chart("size_mb", "Model size, MB (lower is better)")

# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
with tab_arch:
    st.subheader("Seven-layer pipeline (actual processing order)")
    dot = """
    digraph G {
      rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
      l1 [label="Layer 1\\nRequest Intake"]; l2 [label="Layer 2\\nIntent Intelligence"];
      l3 [label="Layer 3\\nCustomer Engagement Scoring"]; l4 [label="Layer 4\\nProfit & Risk Decision Engine"];
      l5 [label="Layer 5\\nDynamic Model Routing"]; slm [label="SLM", fillcolor="#d4edda"];
      llm [label="LLM", fillcolor="#fff3cd"]; l6 [label="Layer 6\\nResponse Generation"];
      l7 [label="Layer 7\\nTelemetry & Financial Monitoring"];
      l1 -> l2 -> l3 -> l4 -> l5; l5 -> slm; l5 -> llm; slm -> l6; llm -> l6; l6 -> l7;
    }"""
    try:
        st.graphviz_chart(dot)
    except Exception:  # noqa: BLE001
        st.code("Query -> Intent -> Scoring -> Profit&Risk -> Routing -> SLM/LLM -> Response -> Telemetry")
    st.markdown(
        "- **Expected Profit** = purchase_probability x cart_value x 0.95 - model_cost - risk_score x 15\n"
        f"- **Route to LLM** if expected profit >= {settings.PROFIT_THRESHOLD} or risk >= {settings.HIGH_RISK_THRESHOLD}; "
        "otherwise SLM\n"
        f"- Simulated costs: SLM = ${settings.SLM_COST}, LLM = ${settings.LLM_COST} per request (configurable)")
