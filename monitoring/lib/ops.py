"""
Module d'opérations statistiques pour le monitoring.

Ce module fournit des fonctions utilitaires pour :
- Calculer des statistiques de latence sur des séries temporelles.
- Calculer le taux d'erreur et le taux de succès à partir des codes de statut HTTP.

Fonctions principales :
- latency_stats_ms : calcule les percentiles p50, p95, p99 et la moyenne des latences.
- error_rate : calcule le pourcentage de requêtes en erreur (codes HTTP >= 400).
- success_rate : calcule le pourcentage de requêtes réussies (codes HTTP == 200).
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def latency_stats_ms(lat: pd.Series) -> Dict[str, float]:
    """
    Calcule les statistiques de latence (en millisecondes) : p50, p95, p99 et moyenne.

    Paramètres
    ----------
    lat : pd.Series
        Série pandas contenant les latences en millisecondes.

    Retourne
    -------
    Dict[str, float]
        Dictionnaire avec les clés 'p50', 'p95', 'p99', 'mean'.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.ops import latency_stats_ms
    >>> lat = pd.Series([100, 200, 300, 400])
    >>> print(latency_stats_ms(lat))
    {'p50': 250.0, 'p95': 385.0, 'p99': 397.0, 'mean': 250.0}
    """
    x = pd.to_numeric(lat, errors="coerce").dropna()
    if x.empty:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "mean": float(x.mean()),
    }


def error_rate(status_code: pd.Series) -> float:
    """
    Calcule le taux d'erreur (codes HTTP >= 400) en pourcentage.

    Paramètres
    ----------
    status_code : pd.Series
        Série pandas contenant les codes de statut HTTP.

    Retourne
    -------
    float
        Pourcentage de requêtes en erreur.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.ops import error_rate
    >>> codes = pd.Series([200, 404, 500, 200])
    >>> print(error_rate(codes))
    50.0
    """
    s = pd.to_numeric(status_code, errors="coerce")
    if s.isna().all():
        return 0.0
    return float((s >= 400).mean()) * 100.0


def success_rate(status_code: pd.Series) -> float:
    """
    Calcule le taux de succès (codes HTTP == 200) en pourcentage.

    Paramètres
    ----------
    status_code : pd.Series
        Série pandas contenant les codes de statut HTTP.

    Retourne
    -------
    float
        Pourcentage de requêtes réussies.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.ops import success_rate
    >>> codes = pd.Series([200, 404, 500, 200])
    >>> print(success_rate(codes))
    50.0
    """
    s = pd.to_numeric(status_code, errors="coerce")
    if s.isna().all():
        return 0.0
    return float((s == 200).mean()) * 100.0