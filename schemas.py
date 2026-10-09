"""Pydantic v2 schemas shared by the API, the pipeline and the dashboard."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Tier = Literal["VIP", "GOLD", "SILVER", "BRONZE"]
Segment = Literal["HOT", "WARM", "COLD"]
RevenueWeight = Literal["HIGH", "MEDIUM", "LOW"]
ModelChoice = Literal["SLM", "LLM"]


class CustomerProfile(BaseModel):
    """Customer engagement data attached to every query."""

    cart_value: float = Field(..., ge=0, le=1_000_000, allow_inf_nan=False,
                              description="Current cart value in dollars (>= 0).")
    tier: Tier = Field(..., description="Loyalty tier: VIP, GOLD, SILVER or BRONZE.")
    page_views_last_10m: int = Field(..., ge=0, le=10_000,
                                     description="Page views in the last 10 minutes (>= 0).")

    @field_validator("tier", mode="before")
    @classmethod
    def _normalise_tier(cls, value: object) -> object:
        """Accept 'gold', ' Gold ' etc."""
        return value.strip().upper() if isinstance(value, str) else value


class IntentResult(BaseModel):
    """Output of the intent classifier."""

    intent: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(default="transformer",
                        description="'transformer' (fine-tuned model) or 'keyword' (fallback).")


class ScoringResult(BaseModel):
    """Customer engagement + intent-risk scoring."""

    segment: Segment
    purchase_probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    revenue_weight: RevenueWeight
    canonical_intent: str = Field(default="unknown")


class RouteDecision(BaseModel):
    """Routing outcome."""

    selected_model: ModelChoice
    expected_profit: float = Field(..., description="Expected profit if the query is escalated to the LLM.")
    model_cost: float = Field(..., ge=0, description="Simulated cost of the selected model.")
    reason: str


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``."""

    query: str = Field(..., min_length=1, max_length=1000)
    profile: CustomerProfile

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: object) -> object:
        """Trim whitespace so that blank queries fail the min_length check."""
        return value.strip() if isinstance(value, str) else value


class ChatResponse(BaseModel):
    """Body returned by ``POST /api/v1/chat``."""

    intent: str
    canonical_intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    classifier_backend: str
    segment: Segment
    purchase_probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    revenue_weight: RevenueWeight
    selected_model: ModelChoice
    expected_profit: float
    model_cost: float
    estimated_savings: float = Field(..., description="Theoretical saving vs. sending this query to the LLM.")
    reason: str
    response: str
    response_source: str
    latency: float = Field(..., ge=0, description="End-to-end gateway latency in milliseconds.")


# Aliases required by the project specification
RoutingDecision = RouteDecision
QueryRequest = ChatRequest
