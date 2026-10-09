import pytest
from fastapi.testclient import TestClient

from main import app

PAYLOAD = {"query": "I want a laptop for gaming",
           "profile": {"cart_value": 1200, "tier": "GOLD", "page_views_last_10m": 8}}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "healthy"}


def test_chat_success(client):
    response = client.post("/api/v1/chat", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    for key in ("intent", "segment", "selected_model", "expected_profit", "response", "confidence",
                "risk_score", "purchase_probability", "latency", "reason"):
        assert key in body
    assert body["selected_model"] in ("SLM", "LLM")
    assert 0.0 <= body["purchase_probability"] <= 1.0


@pytest.mark.parametrize("mutation", [
    {"query": ""}, {"query": "   "},
    {"profile": {"cart_value": -5, "tier": "GOLD", "page_views_last_10m": 1}},
    {"profile": {"cart_value": 5, "tier": "GOLD", "page_views_last_10m": -1}},
    {"profile": {"cart_value": 5, "tier": "PLATINUM", "page_views_last_10m": 1}}])
def test_validation_errors_return_clean_422(client, mutation):
    response = client.post("/api/v1/chat", json={**PAYLOAD, **mutation})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error" and body["details"]


def test_internal_error_is_clean_500(monkeypatch):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        def boom(_request):
            raise RuntimeError("secret internal detail")
        monkeypatch.setattr(app.state.pipeline, "handle", boom)
        response = test_client.post("/api/v1/chat", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert "secret" not in response.text


def test_telemetry_summary_after_chat(client):
    client.post("/api/v1/chat", json=PAYLOAD)
    summary = client.get("/api/v1/telemetry/summary").json()
    assert summary["total_requests"] >= 1
    assert summary["savings_label"] == "Estimated/Theoretical Cost Savings"
