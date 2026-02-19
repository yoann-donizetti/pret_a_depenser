# app/model/predict.py
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def build_X(payload: Dict[str, Any], kept_features: List[str], cat_features: List[str]) -> pd.DataFrame:
    """
    Construit une DataFrame contenant exactement les features attendues par le modèle.

    - 1 seule ligne (un seul client)
    - colonnes dans le même ordre que kept_features
    - valeurs manquantes numériques remplacées par NaN
    - valeurs manquantes catégorielles remplacées par "__MISSING__"
    """

    # 1 ligne, exactement les colonnes attendues
    row = {c: payload.get(c, None) for c in kept_features}
    X = pd.DataFrame([row], columns=kept_features)

    # None -> NaN pour numériques
    X = X.where(pd.notna(X), np.nan)

    # cat_features: str + missing -> "__MISSING__"
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].notna(), "__MISSING__").astype(str)

    return X


def _extract_proba_class1(pred: Any) -> float:
    """
    Extrait la probabilité de la classe 1 (défaut) à partir de différentes formes de sortie.

    Accepte plusieurs formats possibles renvoyés par predict() ou predict_proba() :

    - float                  -> p
    - [p]                    -> p
    - [[p]]                  -> p
    - [[p0, p1]]             -> p1  (probabilité de la classe 1)
    """
    arr = np.asarray(pred)

    if arr.size == 0:
        raise ValueError("Prediction output is empty")

    # cas proba en 2 colonnes
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[0, 1])

    # sinon scalaire
    return float(arr.reshape(-1)[0])


def predict_score(
    model,
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
    threshold: float,
) -> Dict[str, Any]:
    """
    Réalise une prédiction pour un client.

    Retourne :
    - proba_default : probabilité de défaut
    - score : 0 ou 1 selon le seuil
    - decision : ACCEPTED / REFUSED
    """
    X = build_X(payload, kept_features, cat_features)

    # IMPORTANT: ton modèle MLflow pyfunc renvoie actuellement une classe (0/1) via predict().
    # Donc on essaie predict_proba d'abord si dispo, sinon on prend predict() et on vérifie que ça ressemble à une proba.
    if hasattr(model, "predict_proba"):
        proba = _extract_proba_class1(model.predict_proba(X))
    else:
        proba = _extract_proba_class1(model.predict(X))

    # garde-fou: si ça renvoie 0/1, ça passe quand même mais ça devient une "proba" dégénérée
    # (tu l'as vu). On laisse, mais au moins c'est cohérent.
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