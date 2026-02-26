from __future__ import annotations

from typing import Dict

import pandas as pd


def latency_stats_ms(lat: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(lat, errors="coerce").dropna()
    if x.empty:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0,"median": 0.0}
    return {
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "mean": float(x.mean()),
        "median": float(x.median()),
    }


def error_rate(status_code: pd.Series) -> float:
    s = pd.to_numeric(status_code, errors="coerce")
    if s.isna().all():
        return 0.0
    return float((s >= 400).mean()) * 100.0


def success_rate(status_code: pd.Series) -> float:
    s = pd.to_numeric(status_code, errors="coerce")
    if s.isna().all():
        return 0.0
    return float((s == 200).mean()) * 100.0