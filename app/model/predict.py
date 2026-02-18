from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def build_X(payload: Dict[str, Any], kept_features: List[str], cat_features: List[str]) -> pd.DataFrame:
    # 1 ligne, exactement les colonnes attendues
    row = {c: payload.get(c, None) for c in kept_features}
    X = pd.DataFrame([row], columns=kept_features)

    # None -> NaN
    X = X.where(pd.notna(X), np.nan)

    # Catégorielles -> str + "__MISSING__"
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].notna(), "__MISSING__").astype(str)

    return X


def predict_score(
    model,
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    X = build_X(payload, kept_features, cat_features)

    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    proba_default = float(proba[0, 1])

    score = int(proba_default >= threshold)
    decision = "REFUSED" if score == 1 else "ACCEPTED"

    return {
        "SK_ID_CURR": payload.get("SK_ID_CURR"),
        "proba_default": round(proba_default, 6),
        "score": score,
        "decision": decision,
        "threshold": float(threshold),
    }