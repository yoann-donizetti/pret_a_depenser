from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


def apply_time_filter(meta_df: pd.DataFrame, time_window: str) -> pd.DataFrame:
    """
    Filtre sur la colonne 'ts'. Accepte 'all', '24h', '7d', '30d'.
    """
    if meta_df is None or meta_df.empty:
        return meta_df
    if time_window == "all":
        return meta_df

    ts = pd.to_datetime(meta_df["ts"], errors="coerce", utc=True)

    now = datetime.now(timezone.utc)
    if time_window == "24h":
        cutoff = now - timedelta(hours=24)
    elif time_window == "7d":
        cutoff = now - timedelta(days=7)
    elif time_window == "30d":
        cutoff = now - timedelta(days=30)
    else:
        return meta_df

    mask = ts >= cutoff
    return meta_df.loc[mask].copy()


def filter_rows_by_meta_ts(rows: list[dict], meta_df: pd.DataFrame) -> list[dict]:
    """
    Refiltre rows pour rester cohérent avec meta filtrée (ts).
    """
    if not rows or meta_df is None or meta_df.empty:
        return []

    kept = set(pd.to_datetime(meta_df["ts"], errors="coerce", utc=True).astype(str).tolist())
    out = []
    for r in rows:
        ts = pd.to_datetime(r.get("ts"), errors="coerce", utc=True)
        if str(ts) in kept:
            out.append(r)
    return out