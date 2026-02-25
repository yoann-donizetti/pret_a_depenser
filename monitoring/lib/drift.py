from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def psi_from_dists(ref_p: np.ndarray, prod_p: np.ndarray, eps: float = 1e-6) -> float:
    r = np.clip(ref_p, eps, 1)
    p = np.clip(prod_p, eps, 1)
    return float(np.sum((p - r) * np.log(p / r)))


def prod_dist_numeric(prod_s: pd.Series, edges: List[float], labels: List[str]) -> Tuple[List[str], np.ndarray]:
    x = pd.to_numeric(prod_s, errors="coerce").dropna()
    if len(x) == 0:
        return labels, np.zeros(len(labels), dtype=float)

    b = pd.cut(x, bins=np.array(edges, dtype=float), include_lowest=True, duplicates="drop")
    dist = b.value_counts(normalize=True).sort_index()
    prod_map = {str(idx): float(v) for idx, v in dist.items()}

    p = np.array([prod_map.get(lab, 0.0) for lab in labels], dtype=float)
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


def count_drift(drift_df: pd.DataFrame, threshold: float = 0.25) -> int:
    if drift_df.empty or "psi" not in drift_df.columns:
        return 0
    x = pd.to_numeric(drift_df["psi"], errors="coerce")
    return int((x > threshold).sum())