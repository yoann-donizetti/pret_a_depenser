from __future__ import annotations

TIME_WINDOWS = ["all", "24h", "7d", "30d"]

PSI_THRESHOLDS = {
    "ok": 0.10,
    "watch": 0.25,
}

DEFAULTS = {
    "endpoint": "/predict",
    "limit": 1000,
    "p95_threshold_ms": 200,
    "drift_threshold": 0.25,
    "max_limit_fetch": 1_000_000,
    "topk_drift": 20,
}