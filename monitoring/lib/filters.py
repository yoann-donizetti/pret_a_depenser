"""
Module de filtres pour le monitoring.

Ce module fournit des fonctions utilitaires pour :
- Filtrer un DataFrame de métadonnées selon une fenêtre temporelle (24h, 7d, 30d, all).
- Refiltrer une liste de requêtes pour rester cohérent avec un DataFrame filtré sur les timestamps.
fonctions principales : 
- apply_time_filter : filtre un DataFrame de métadonnées selon une fenêtre temporelle.
- filter_rows_by_meta_ts : refiltre une liste de requêtes pour rester cohérent avec un DataFrame de métadonnées filtré sur les timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


def apply_time_filter(meta_df: pd.DataFrame, time_window: str) -> pd.DataFrame:
    """
    Filtre un DataFrame sur la colonne 'ts' selon une fenêtre temporelle.
    Accepte 'all', '24h', '7d', '30d'.

    Paramètres
    ----------
    meta_df : pd.DataFrame
        DataFrame contenant une colonne 'ts' (timestamps).
    time_window : str
        Fenêtre temporelle à appliquer ('all', '24h', '7d', '30d').

    Retourne
    -------
    pd.DataFrame
        DataFrame filtré selon la fenêtre temporelle.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.filters import apply_time_filter
    >>> df = pd.DataFrame({'ts': ['2024-02-01T12:00:00Z', '2024-02-20T12:00:00Z']})
    >>> df_filtre = apply_time_filter(df, time_window="7d")
    >>> print(df_filtre)
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
    Refiltre une liste de requêtes (rows) pour rester cohérent avec un DataFrame de métadonnées filtré sur les timestamps.

    Paramètres
    ----------
    rows : list[dict]
        Liste de dictionnaires représentant les requêtes (doivent contenir une clé 'ts').
    meta_df : pd.DataFrame
        DataFrame filtré contenant la colonne 'ts'.

    Retourne
    -------
    list[dict]
        Liste filtrée de requêtes, cohérente avec les timestamps du DataFrame.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.filters import filter_rows_by_meta_ts
    >>> rows = [{'ts': '2024-02-01T12:00:00Z'}, {'ts': '2024-02-20T12:00:00Z'}]
    >>> meta_df = pd.DataFrame({'ts': ['2024-02-20T12:00:00Z']})
    >>> rows_filtrees = filter_rows_by_meta_ts(rows, meta_df)
    >>> print(rows_filtrees)
    [{'ts': '2024-02-20T12:00:00Z'}]
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