"""Layer 7 - Telemetry & financial monitoring.

All savings are *theoretical*: they compare the simulated model costs of the
actual routing against a baseline that sends 100% of traffic to the LLM.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

from config import TELEMETRY_LOG_PATH, Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """One routed request (no query text is stored, for privacy)."""

    timestamp: str
    intent: str
    segment: str
    selected_model: str
    expected_profit: float
    model_cost: float
    estimated_savings: float
    latency_ms: float


class Telemetry:
    """Thread-safe in-memory metrics store with optional JSONL persistence."""

    LABEL = "Estimated/Theoretical Cost Savings"

    def __init__(self, settings: Optional[Settings] = None, persist_path: Optional[Path] = None,
                 max_records: int = 10_000) -> None:
        settings = settings or get_settings()
        self.slm_cost = float(settings.SLM_COST)
        self.llm_cost = float(settings.LLM_COST)
        self.persist_path: Optional[Path] = (
            (persist_path or TELEMETRY_LOG_PATH) if settings.TELEMETRY_PERSIST or persist_path else None
        )
        self._records: Deque[TelemetryRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, intent: str, segment: str, selected_model: str, expected_profit: float,
               model_cost: float, latency_ms: float) -> TelemetryRecord:
        """Store one request and return the stored record."""
        entry = TelemetryRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            intent=intent, segment=segment, selected_model=selected_model,
            expected_profit=round(float(expected_profit), 4), model_cost=float(model_cost),
            estimated_savings=round(self.llm_cost - float(model_cost), 6),
            latency_ms=round(float(latency_ms), 3),
        )
        with self._lock:
            self._records.append(entry)
            if self.persist_path is not None:
                try:
                    self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.persist_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(entry)) + "\n")
                except OSError:
                    logger.warning("Could not write telemetry log file.")
        return entry

    def summary(self) -> Dict[str, object]:
        """Aggregate statistics for the dashboard and ``/api/v1/telemetry/summary``."""
        with self._lock:
            records = list(self._records)
        total = len(records)
        llm_n = sum(1 for r in records if r.selected_model == "LLM")
        slm_n = total - llm_n
        baseline = total * self.llm_cost
        actual = sum(r.model_cost for r in records)
        savings = baseline - actual
        return {
            "total_requests": total,
            "slm_requests": slm_n,
            "llm_requests": llm_n,
            "llm_share_percent": round(100.0 * llm_n / total, 2) if total else 0.0,
            "avg_latency_ms": round(sum(r.latency_ms for r in records) / total, 3) if total else 0.0,
            "total_expected_profit": round(sum(r.expected_profit for r in records), 4),
            "llm_baseline_cost": round(baseline, 6),
            "actual_cost": round(actual, 6),
            "estimated_cost_savings": round(savings, 6),
            "estimated_savings_percent": round(100.0 * savings / baseline, 2) if baseline > 0 else 0.0,
            "savings_label": self.LABEL,
        }

    def recent(self, n: int = 20) -> List[Dict[str, object]]:
        """Return the ``n`` most recent records (newest last)."""
        with self._lock:
            return [asdict(r) for r in list(self._records)[-max(n, 0):]]
