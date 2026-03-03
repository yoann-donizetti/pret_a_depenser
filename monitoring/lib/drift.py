from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def psi_from_dists(ref_p: np.ndarray, prod_p: np.ndarray, eps: float = 1e-6) -> float:
    r = np.clip(ref_p, eps, 1)
    p = np.clip(prod_p, eps, 1)
    return float(np.sum((p - r) * np.log(p / r)))


from typing import List, Tuple
import numpy as np
import pandas as pd


def prod_dist_numeric(prod_s: pd.Series, edges: List[float], labels: List[str]) -> Tuple[List[str], np.ndarray]:
    x = pd.to_numeric(prod_s, errors="coerce").dropna()
    if len(x) == 0:
        return labels, np.zeros(len(labels), dtype=float)

    # IMPORTANT: si labels est fourni, on les impose à pd.cut
    b = pd.cut(
        x,
        bins=np.array(edges, dtype=float),
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    # value_counts -> index = labels (donc "0-1", "1-2", etc.)
    dist = b.value_counts(normalize=True).reindex(labels).fillna(0.0)

    p = dist.to_numpy(dtype=float)

    # sécurité (si arrondis/NaN)
    s = float(p.sum())
    if s > 0:
        p = p / s

    return labels, p


def prod_dist_categorical(prod_s: pd.Series, labels_ref: List[str]) -> Tuple[List[str], np.ndarray]:
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
    Construit une table: feature | psi | type
    """
    if prod_inputs is None or prod_inputs.empty or not ref_rows:
        return pd.DataFrame(columns=["feature", "psi", "type"])

    ref_map = {r["feature"]: r for r in ref_rows}
    ref_features = [r["feature"] for r in ref_rows if r.get("feature") not in excluded_features]

    common = [c for c in ref_features if c in prod_inputs.columns]

    psi_rows: List[Dict[str, Any]] = []
    for feat in common:
        if feat in excluded_features:
            continue

        ref = ref_map[feat]
        kind = ref.get("kind")
        ref_dist = ref.get("ref_dist_json") or {}
        labels_ref = ref_dist.get("labels") or []
        ref_p = np.array(ref_dist.get("p") or [], dtype=float)

        prod_s = prod_inputs[feat]

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
    if drift_df is None or drift_df.empty or "psi" not in drift_df.columns:
        return 0
    return int((drift_df["psi"] > threshold).sum())