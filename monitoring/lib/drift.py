"""
Module d'analyse de la dérive des variables (drift) pour le monitoring.

Ce module fournit des fonctions pour :
- Calculer le Population Stability Index (PSI) entre distributions de référence et de production.
- Générer les distributions de production pour les variables numériques et catégorielles.
- Construire une table synthétique du drift pour toutes les variables suivies.
- Compter le nombre de variables en dérive selon un seuil donné.

"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def psi_from_dists(ref_p: np.ndarray, prod_p: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calcule le Population Stability Index (PSI) entre deux distributions.

    Paramètres
    ----------
    ref_p : np.ndarray
        Distribution de référence (somme = 1).
    prod_p : np.ndarray
        Distribution de production (somme = 1).
    eps : float, optionnel
        Valeur minimale pour éviter les divisions par zéro (par défaut 1e-6).

    Retourne
    -------
    float
        Valeur du PSI.

    Exemple
    -------
    >>> import numpy as np
    >>> from monitoring.lib.drift import psi_from_dists
    >>> ref = np.array([0.7, 0.3])
    >>> prod = np.array([0.5, 0.5])
    >>> psi = psi_from_dists(ref, prod)
    >>> print(round(psi, 3))
    0.091
    """
    r = np.clip(ref_p, eps, 1)
    p = np.clip(prod_p, eps, 1)
    return float(np.sum((p - r) * np.log(p / r)))


def prod_dist_numeric(prod_s: pd.Series, edges: List[float], labels: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Calcule la distribution de production pour une variable numérique selon les intervalles de référence.

    Paramètres
    ----------
    prod_s : pd.Series
        Série pandas contenant les valeurs de production.
    edges : List[float]
        Bornes des intervalles (bins) utilisées pour le découpage.
    labels : List[str]
        Labels associés à chaque intervalle.

    Retourne
    -------
    Tuple[List[str], np.ndarray]
        Labels des intervalles et distribution normalisée (somme = 1).

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.drift import prod_dist_numeric
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> labels, p = prod_dist_numeric(s, edges=[0, 2, 4, 6], labels=["[0,2[", "[2,4[", "[4,6]"])
    >>> print(labels)
    ['[0,2[', '[2,4[', '[4,6]']
    >>> print(p.round(2))
    [0.2 0.4 0.4]
    """
    x = pd.to_numeric(prod_s, errors="coerce").dropna()
    if len(x) == 0:
        return labels, np.zeros(len(labels), dtype=float)

    b = pd.cut(
        x,
        bins=np.array(edges, dtype=float),
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    dist = b.value_counts(normalize=True).reindex(labels).fillna(0.0)
    p = dist.to_numpy(dtype=float)
    s = float(p.sum())
    if s > 0:
        p = p / s

    return labels, p


def prod_dist_categorical(prod_s: pd.Series, labels_ref: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Calcule la distribution de production pour une variable catégorielle selon les labels de référence.

    Paramètres
    ----------
    prod_s : pd.Series
        Série pandas contenant les valeurs de production.
    labels_ref : List[str]
        Liste des labels de référence (modalités attendues).

    Retourne
    -------
    Tuple[List[str], np.ndarray]
        Labels de référence et distribution normalisée (somme = 1).

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.drift import prod_dist_categorical
    >>> s = pd.Series(["A", "B", "A", "C", None])
    >>> labels, p = prod_dist_categorical(s, labels_ref=["A", "B", "C", "__MISSING__"])
    >>> print(labels)
    ['A', 'B', 'C', '__MISSING__']
    >>> print(p.round(2))
    [0.4 0.2 0.2 0.2]
    """
    x = prod_s.fillna("__MISSING__").astype(str)

    has_other = "__OTHER__" in labels_ref
    allowed = set(labels_ref)

    mapped = []
    for v in x.tolist():
        if v in allowed:
            mapped.append(v)
        else:
            mapped.append("__OTHER__" if has_other else v)

    vc = pd.Series(mapped).value_counts(normalize=True)
    prod_map = {k: float(v) for k, v in vc.items()}

    p = np.array([prod_map.get(lab, 0.0) for lab in labels_ref], dtype=float)
    return labels_ref, p


def compute_drift_table(
    *,
    prod_inputs: pd.DataFrame,
    ref_rows: List[Dict[str, Any]],
    excluded_features: set[str],
) -> pd.DataFrame:
    """
    Construit une table synthétique du drift pour chaque variable suivie (feature, psi, type).

    Paramètres
    ----------
    prod_inputs : pd.DataFrame
        Données de production (features d'entrée).
    ref_rows : List[Dict[str, Any]]
        Liste des distributions de référence pour chaque feature.
    excluded_features : set[str]
        Ensemble des features à exclure de l'analyse.

    Retourne
    -------
    pd.DataFrame
        Table contenant les colonnes : feature, psi, type.

    Exemple
    -------
    >>> from monitoring.lib.drift import compute_drift_table
    >>> drift_df = compute_drift_table(prod.inputs=inputs, ref_rows=ref, excluded_features={"user_id"})
    >>> print(drift_df.head())
    """
    if prod.inputs is None or prod.inputs.empty or not ref_rows:
        return pd.DataFrame(columns=["feature", "psi", "type"])

    ref_map = {r["feature"]: r for r in ref_rows}
    ref_features = [r["feature"] for r in ref_rows if r.get("feature") not in excluded_features]

    common = [c for c in ref_features if c in prod.inputs.columns]

    psi_rows: List[Dict[str, Any]] = []
    for feat in common:
        if feat in excluded_features:
            continue

        ref = ref_map[feat]
        kind = ref.get("kind")
        ref_dist = ref.get("ref_dist_json") or {}
        labels_ref = ref_dist.get("labels") or []
        ref_p = np.array(ref_dist.get("p") or [], dtype=float)

        prod_s = prod.inputs[feat]

        v = np.nan
        if kind == "numeric":
            bins = ref.get("bins_json") or {}
            edges = bins.get("edges") or []
            if edges and labels_ref and ref_p.size:
                _, prod_p = prod_dist_numeric(prod_s, edges=edges, labels=labels_ref)
                v = psi_from_dists(ref_p, prod_p)
        else:
            if labels_ref and ref_p.size:
                _, prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref)
                v = psi_from_dists(ref_p, prod_p)

        psi_rows.append({"feature": feat, "psi": v, "type": kind})

    drift = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
    return drift


def count_drift(drift_df: pd.DataFrame, threshold: float = 0.25) -> int:
    """
    Compte le nombre de variables dont le PSI dépasse un seuil donné (drift détecté).

    Paramètres
    ----------
    drift_df : pd.DataFrame
        Table du drift (doit contenir une colonne 'psi').
    threshold : float, optionnel
        Seuil à partir duquel on considère qu'il y a drift (par défaut 0.25).

    Retourne
    -------
    int
        Nombre de variables en drift.

    Exemple
    -------
    >>> from monitoring.lib.drift import count_drift
    >>> n = count_drift(drift_df, threshold=0.2)
    >>> print(n)
    """
    if drift_df is None or drift_df.empty or "psi" not in drift_df.columns:
        return 0
    return int((drift_df["psi"] > threshold).sum())