"""Layer 6 - Response generation for the SLM and LLM paths.

* SLM path: fast template answers chosen by the intent classifier's output.
* LLM path: ``generate_llm_response``. By default a deterministic local mock is
  used, so no API key or internet is needed. Set ``LLM_PROVIDER`` to ``ollama``,
  ``openai`` or ``groq`` (plus ``LLM_API_KEY`` where needed) to use a real model.
  Any provider error falls back to the mock so the gateway keeps working.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import requests

from config import Settings, get_settings
from scoring_engine import normalize_intent_label

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = ("You are a helpful, concise e-commerce customer-support assistant. "
                 "Do not claim to have performed actions you cannot perform.")

_SLM_TEMPLATES = {
    "payment_issue": "Sorry about the payment trouble. Please check your card details and try again, "
                     "or use another payment method.",
    "returns_refunds": "You can start a return from the Orders page. Refunds usually reach your original "
                       "payment method within a few business days after we receive the item.",
    "product_recommendation": "Happy to help you choose! Tell me your budget and what you will use it for "
                              "and I will narrow the options down.",
    "order_status": "You can follow your order from the Orders page using your tracking number.",
    "general_inquiry": "Thanks for reaching out! How can I help you today?",
    "unknown": "Thanks for your message. I can help with orders, payments, returns and product advice.",
}

_LLM_MOCK_TEMPLATES = {
    "payment_issue": "I'm sorry your payment didn't go through - that's frustrating. Common causes are an "
                     "expired card, a billing-address mismatch or a bank security hold. Please re-check the "
                     "card details, try a different payment method, or contact your bank. If it keeps failing, "
                     "reply here and a support specialist will review the transaction with you.",
    "returns_refunds": "I can help with that return. Open the order under 'My Orders', choose 'Return item' and "
                       "pick a reason. Once the item is received and inspected, the refund is issued to the "
                       "original payment method. If the item is outside the return window, tell me and I will "
                       "explain the available options.",
    "product_recommendation": "Great choice to explore! To recommend the best fit I would compare your budget, "
                              "main use case and must-have features. Share those and I will suggest two or three "
                              "options with the trade-offs of each, plus accessories that pair well with them.",
    "order_status": "Let's find your order. Use the tracking link in your confirmation email or the 'My Orders' "
                    "page for the latest status. If it has not moved for several days, send me the order number "
                    "and I will help you investigate.",
    "general_inquiry": "Thanks for getting in touch! I can help with orders, payments, returns and product "
                       "advice. Tell me a little more about what you need and I will point you in the right "
                       "direction.",
    "unknown": "Thanks for your message. I want to make sure I get this right - could you share a few more "
               "details about what you need help with?",
}


class LLMService:
    """Produces customer-facing responses for both routing paths."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.provider = self.settings.LLM_PROVIDER.strip().lower()

    # ---------------------------- SLM path ---------------------------- #
    def generate_slm_response(self, query: str, intent: Optional[str] = None) -> str:
        """Template answer for the cheap path (``query`` is unused by templates)."""
        canonical = normalize_intent_label(intent or "") or "unknown"
        return _SLM_TEMPLATES[canonical]

    # ---------------------------- LLM path ---------------------------- #
    def generate_llm_response(self, query: str, intent: Optional[str] = None) -> str:
        """Answer with the configured LLM provider (mock by default)."""
        return self._generate_llm(query, intent)[0]

    def _generate_llm(self, query: str, intent: Optional[str]) -> Tuple[str, str]:
        if self.provider in {"ollama", "openai", "groq"}:
            try:
                return self._call_provider(query), f"llm_{self.provider}"
            except Exception as exc:  # noqa: BLE001 - never break the gateway
                logger.warning("LLM provider '%s' failed (%s). Using local mock.", self.provider,
                               type(exc).__name__)
        elif self.provider != "mock":
            logger.warning("Unknown LLM_PROVIDER '%s'. Using local mock.", self.provider)
        return self._mock_response(intent), "llm_mock"

    @staticmethod
    def _mock_response(intent: Optional[str]) -> str:
        canonical = normalize_intent_label(intent or "") or "unknown"
        return _LLM_MOCK_TEMPLATES[canonical]

    def _call_provider(self, query: str) -> str:
        s = self.settings
        timeout = s.LLM_TIMEOUT_SECONDS
        if self.provider == "ollama":
            base = (s.LLM_BASE_URL or "http://localhost:11434").rstrip("/")
            resp = requests.post(f"{base}/api/generate", timeout=timeout, json={
                "model": s.LLM_MODEL_NAME, "stream": False,
                "prompt": f"{SYSTEM_PROMPT}\n\nCustomer: {query}\nAssistant:"})
            resp.raise_for_status()
            return str(resp.json()["response"]).strip()

        default_base = "https://api.groq.com/openai/v1" if self.provider == "groq" else "https://api.openai.com/v1"
        base = (s.LLM_BASE_URL or default_base).rstrip("/")
        if not s.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is not set")
        resp = requests.post(f"{base}/chat/completions", timeout=timeout,
                             headers={"Authorization": f"Bearer {s.LLM_API_KEY}"},
                             json={"model": s.LLM_MODEL_NAME,
                                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                {"role": "user", "content": query}]})
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()

    # ----------------------------- router ----------------------------- #
    def respond(self, selected_model: str, query: str, intent: Optional[str]) -> Tuple[str, str]:
        """Return ``(response_text, response_source)`` for the chosen model."""
        if selected_model == "LLM":
            return self._generate_llm(query, intent)
        return self.generate_slm_response(query, intent), "slm_template"


def generate_llm_response(query: str, intent: Optional[str] = None) -> str:
    """Module-level convenience wrapper around :class:`LLMService`."""
    return LLMService().generate_llm_response(query, intent)
