from __future__ import annotations

from typing import Dict

import pandas as pd


TIMING_COLS = ["db_ms", "validation_ms", "inference_ms", "total_ms"]


def extract_timings(outputs_df: pd.DataFrame) -> pd.DataFrame:
    """
    outputs['timing'] est un dict JSON -> on le normalise.
    """
    if outputs_df is None or outputs_df.empty or "timing" not in outputs_df.columns:
        return pd.DataFrame()

    s = outputs_df["timing"].dropna()
    if s.empty:
        return pd.DataFrame()

    df = pd.json_normalize(s)
    for c in TIMING_COLS:
        if c not in df.columns:
            df[c] = 0.0

    out = df[TIMING_COLS].copy()
    for c in TIMING_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def series_stats_ms(s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "mean": float(x.mean()),
    }


def compute_timing_stats(timing_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if timing_df is None or timing_df.empty:
        return {}
    return {c: series_stats_ms(timing_df[c]) for c in TIMING_COLS if c in timing_df.columns}