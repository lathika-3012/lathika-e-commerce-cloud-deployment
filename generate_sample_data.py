"""Generate sample queries + customer profiles and (optionally) run them through the gateway.

    python generate_sample_data.py            # writes data/processed/sample_requests.json
    python generate_sample_data.py --run      # also routes every sample locally and prints a table
    python generate_sample_data.py --api      # sends every sample to the running FastAPI server
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from config import SAMPLE_REQUESTS_PATH, ensure_directories, get_settings

SAMPLES: List[Dict[str, object]] = [
    {"query": "I want to buy a laptop for work", "profile": {"cart_value": 1200, "tier": "GOLD", "page_views_last_10m": 8}},
    {"query": "My payment failed", "profile": {"cart_value": 80, "tier": "BRONZE", "page_views_last_10m": 2}},
    {"query": "Where is my order?", "profile": {"cart_value": 0, "tier": "SILVER", "page_views_last_10m": 1}},
    {"query": "I want to return this product", "profile": {"cart_value": 45, "tier": "SILVER", "page_views_last_10m": 3}},
    {"query": "Can you recommend headphones?", "profile": {"cart_value": 250, "tier": "VIP", "page_views_last_10m": 15}},
    {"query": "Hello, what are your opening hours?", "profile": {"cart_value": 0, "tier": "BRONZE", "page_views_last_10m": 0}},
    {"query": "Please suggest a gift under $30", "profile": {"cart_value": 12, "tier": "BRONZE", "page_views_last_10m": 1}},
    {"query": "I was charged twice for my order", "profile": {"cart_value": 0, "tier": "GOLD", "page_views_last_10m": 4}},
    {"query": "How do I get a refund?", "profile": {"cart_value": 20, "tier": "BRONZE", "page_views_last_10m": 2}},
    {"query": "Track my package please", "profile": {"cart_value": 3, "tier": "BRONZE", "page_views_last_10m": 0}},
    {"query": "Which gaming laptop is best for students?", "profile": {"cart_value": 2200, "tier": "VIP", "page_views_last_10m": 20}},
    {"query": "I have a question about your newsletter", "profile": {"cart_value": 0, "tier": "SILVER", "page_views_last_10m": 0}},
]


def write_samples() -> None:
    """Write the sample file."""
    ensure_directories()
    SAMPLE_REQUESTS_PATH.write_text(json.dumps(SAMPLES, indent=2), encoding="utf-8")
    print(f"Wrote {len(SAMPLES)} sample requests to {SAMPLE_REQUESTS_PATH}")


def _print_row(query: str, data: Dict[str, object]) -> None:
    print(f"{query[:40]:<41}{str(data['intent'])[:24]:<25}{data['segment']:<6}"
          f"{float(data['risk_score']):<6.2f}{float(data['expected_profit']):>10.2f}  {data['selected_model']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="route samples with the local pipeline")
    parser.add_argument("--api", action="store_true", help="route samples through the FastAPI server")
    args = parser.parse_args()
    write_samples()
    if not (args.run or args.api):
        return 0

    print(f"\n{'QUERY':<41}{'INTENT':<25}{'SEG':<6}{'RISK':<6}{'PROFIT':>10}  MODEL")
    if args.api:
        import requests

        url = get_settings().API_URL.rstrip("/") + "/api/v1/chat"
        for sample in SAMPLES:
            try:
                resp = requests.post(url, json=sample, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"ERROR calling {url}: {exc}")
                return 1
            _print_row(str(sample["query"]), resp.json())
    else:
        from pipeline import GatewayPipeline
        from schemas import ChatRequest

        pipeline = GatewayPipeline()
        for sample in SAMPLES:
            result = pipeline.handle(ChatRequest(**sample))
            _print_row(str(sample["query"]), result.model_dump())
        summary = pipeline.telemetry.summary()
        print(f"\n{summary['savings_label']}: ${summary['estimated_cost_savings']:.4f} "
              f"({summary['estimated_savings_percent']}% vs. 100% LLM) over {summary['total_requests']} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
