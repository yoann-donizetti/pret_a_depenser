from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


def apply_time_filter(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    window: "all" | "24h" | "7d" | "30d"
    df doit contenir une colonne "ts" parseable.
    """
    if df.empty or "ts" not in df.columns:
        return df

    if window == "all":
        return df

    # Assure datetime
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    out = df.copy()
    out["ts"] = ts

    now = datetime.now(timezone.utc)

    if window == "24h":
        cutoff = now - timedelta(hours=24)
    elif window == "7d":
        cutoff = now - timedelta(days=7)
    elif window == "30d":
        cutoff = now - timedelta(days=30)
    else:
        return out

    return out[out["ts"].notna() & (out["ts"] >= cutoff)]