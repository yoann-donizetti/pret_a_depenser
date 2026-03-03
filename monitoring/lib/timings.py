
"""
Module de gestion et d'analyse des temps d'exécution (timings) pour les différentes étapes d'un pipeline de traitement.

Ce module fournit des fonctions pour extraire, normaliser et calculer des statistiques sur les temps d'exécution stockés dans des DataFrames pandas.
Il permet notamment d'obtenir des métriques comme la médiane (p50), les percentiles (p95, p99) et la moyenne pour chaque étape (accès base de données, validation, inférence, total).

Exemples d'utilisation :
    - Extraire les timings d'une colonne 'timing' d'un DataFrame
    - Calculer des statistiques descriptives sur les temps d'exécution

Fonctions principales :
    - extract_timings(outputs_df): Extrait et normalise les colonnes de temps à partir d'une colonne 'timing' contenant des dictionnaires JSON.
    - series_stats_ms(s): Calcule des statistiques de base (p50, p95, p99, moyenne) sur une série de temps en millisecondes.
    - compute_timing_stats(timing_df): Calcule les statistiques de temps pour chaque colonne de temps d'un DataFrame.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


TIMING_COLS = ["db_ms", "validation_ms", "inference_ms", "total_ms"]


def extract_timings(outputs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait et normalise les colonnes de temps à partir d'une colonne 'timing' contenant des dictionnaires JSON dans un DataFrame.

    Args:
        outputs_df (pd.DataFrame): DataFrame contenant une colonne 'timing' avec des dictionnaires de temps.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes ['db_ms', 'validation_ms', 'inference_ms', 'total_ms'] converties en float.

    Exemple :
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'timing': [
        ...         {'db_ms': 10, 'validation_ms': 5, 'inference_ms': 20, 'total_ms': 35},
        ...         {'db_ms': 12, 'validation_ms': 6, 'inference_ms': 18, 'total_ms': 36}
        ...     ]
        ... })
        >>> extract_timings(df)
           db_ms  validation_ms  inference_ms  total_ms
        0   10.0            5.0          20.0      35.0
        1   12.0            6.0          18.0      36.0
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
    """
    Calcule des statistiques de base (p50, p95, p99, moyenne) sur une série de temps en millisecondes.

    Args:
        s (pd.Series): Série de valeurs numériques (temps en ms).

    Returns:
        Dict[str, float]: Dictionnaire avec les clés 'p50', 'p95', 'p99', 'mean'.

    Exemple :
        >>> import pandas as pd
        >>> s = pd.Series([10, 20, 30, 40, 50])
        >>> series_stats_ms(s)
        {'p50': 30.0, 'p95': 48.0, 'p99': 49.6, 'mean': 30.0}
    """
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
    """
    Calcule les statistiques de temps (p50, p95, p99, moyenne) pour chaque colonne de temps d'un DataFrame.

    Args:
        timing_df (pd.DataFrame): DataFrame contenant les colonnes de temps.

    Returns:
        Dict[str, Dict[str, float]]: Dictionnaire de statistiques pour chaque colonne de temps.

    Exemple :
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'db_ms': [10, 12],
        ...     'validation_ms': [5, 6],
        ...     'inference_ms': [20, 18],
        ...     'total_ms': [35, 36]
        ... })
        >>> compute_timing_stats(df)
        {'db_ms': {'p50': 11.0, 'p95': 11.9, 'p99': 11.98, 'mean': 11.0}, ...}
    """
    if timing_df is None or timing_df.empty:
        return {}
    return {c: series_stats_ms(timing_df[c]) for c in TIMING_COLS if c in timing_df.columns}