"""
Module de gestion et chargement des données pour le monitoring.

Ce module fournit des fonctions utilitaires pour :
- Charger et filtrer les données de production issues des requêtes API (inputs, outputs, métadonnées).
- Récupérer les distributions de référence des features depuis la base de données.
- Exclure certaines colonnes sensibles ou inutiles des jeux de données.

Fonctions principales :
- load_prod_data : charge les données de production pour un endpoint donné, avec filtrage temporel et exclusion de colonnes.
- load_reference : récupère toutes les distributions de référence des features.
- load_reference_one : récupère la distribution de référence d'un feature spécifique.

Ces fonctions sont utilisées pour l'analyse de la dérive, la comparaison entre production et référence, et la visualisation des distributions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from core.db.repo_prod_requests import select_prod_requests
from core.db.repo_ref_dist import load_all_ref, load_one_ref

from monitoring.lib.filters import apply_time_filter, filter_rows_by_meta_ts
from monitoring.lib.security import drop_excluded_columns


def load_prod_data(
    *,
    endpoint: str,
    limit: int | None,
    time_window: str,
    excluded_features: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Charge et filtre les données de production pour un endpoint donné.

    Paramètres
    ----------
    endpoint : str
        Nom de l'endpoint à interroger.
    limit : int | None
        Nombre maximum de requêtes à charger (None pour tout charger).
    time_window : str
        Fenêtre temporelle à appliquer pour filtrer les données (ex: '7d', '30d').
    excluded_features : set[str]
        Ensemble des noms de colonnes à exclure des inputs.

    Retourne
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]
        - prod_meta_df : DataFrame des métadonnées (timestamp, endpoint, status_code, etc.)
        - prod_inputs_df : DataFrame des features d'entrée (inputs)
        - prod_outputs_df : DataFrame des features de sortie (outputs)
        - raw_rows_filtered : Liste des requêtes filtrées (dictionnaires bruts)

    Exemple
    -------
    >>> from monitoring.lib.data import load_prod_data
    >>> meta, inputs, outputs, rows = load_prod_data(
    ...     endpoint="predict",
    ...     limit=1000,
    ...     time_window="7d",
    ...     excluded_features={"user_id", "ip"}
    ... )
    >>> print(meta.head())
    >>> print(inputs.head())
    >>> print(outputs.head())
    """
    rows = select_prod_requests(endpoint=endpoint, limit=(limit or 1_000_000))
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    prod_meta = pd.DataFrame(
        [{k: r.get(k) for k in ["ts", "endpoint", "status_code", "latency_ms", "error", "message"]} for r in rows]
    )

    prod_meta = apply_time_filter(prod_meta, time_window)
    rows = filter_rows_by_meta_ts(rows, prod_meta)

    prod_inputs = pd.DataFrame([r.get("inputs") or {} for r in rows])
    prod_outputs = pd.DataFrame([r.get("outputs") or {} for r in rows])

    prod_inputs = drop_excluded_columns(prod_inputs, excluded_features)

    return prod_meta, prod_inputs, prod_outputs, rows


def load_reference() -> List[Dict]:
    """
    Charge toutes les distributions de référence des features depuis la base de données.

    Retourne
    -------
    List[Dict]
        Liste de dictionnaires représentant les distributions de référence pour chaque feature.

    Exemple
    -------
    >>> from monitoring.lib.data import load_reference
    >>> ref = load_reference()
    >>> print(ref[0])  # Affiche la distribution de référence du premier feature
    """
    return load_all_ref()


def load_reference_one(feature: str) -> Dict | None:
    """
    Charge la distribution de référence d'un feature donné depuis la base de données.

    Paramètres
    ----------
    feature : str
        Nom du feature dont on souhaite récupérer la distribution de référence.

    Retourne
    -------
    Dict | None
        Dictionnaire représentant la distribution de référence du feature, ou None si non trouvé.

    Exemple
    -------
    >>> from monitoring.lib.data import load_reference_one
    >>> ref = load_reference_one("age")
    >>> print(ref)
    """
    return load_one_ref(feature)