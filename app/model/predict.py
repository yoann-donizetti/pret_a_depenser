from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def build_X(payload: Dict[str, Any], kept_features: List[str], cat_features: List[str]) -> pd.DataFrame:
    row = {c: payload.get(c, None) for c in kept_features}
    X = pd.DataFrame([row], columns=kept_features)

    # None -> NaN
    X = X.where(pd.notna(X), np.nan)

    # cat missing -> "__MISSING__"
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].notna(), "__MISSING__").astype(str)

    return X


def _extract_proba_class1(pred: Any) -> float:
    arr = np.asarray(pred)
    if arr.size == 0:
        raise ValueError("Prediction output is empty")

    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[0, 1])

    return float(arr.reshape(-1)[0])


def predict_score(
    model,
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
    threshold: float,
) -> Dict[str, Any]:
    X = build_X(payload, kept_features, cat_features)

    if hasattr(model, "predict_proba"):
        proba = _extract_proba_class1(model.predict_proba(X))
    else:
        proba = _extract_proba_class1(model.predict(X))

    proba = float(proba)
    score = int(proba >= float(threshold))
    decision = "REFUSED" if score == 1 else "ACCEPTED"

    return {
        "SK_ID_CURR": payload.get("SK_ID_CURR", None),
        "proba_default": round(proba, 6),
        "score": score,
        "decision": decision,
        "threshold": float(threshold),
    }