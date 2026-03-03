
"""
Module de prédiction pour le modèle CatBoost :
 - Construction optimisée de la ligne d'entrée pour le modèle
 - Gestion des features catégorielles et des valeurs manquantes
 - Prédiction du score et de la décision (ACCEPTED/REFUSED)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import inspect
import numpy as np





class _MiniLoc:
    """
    Accès simplifié aux valeurs d'une MiniFrame par index et nom de colonne (simule .loc de pandas).
    """
    def __init__(self, frame: "MiniFrame"):
        self._f = frame

    def __getitem__(self, key):
        # key attendu: (row_idx, col_name)
        i, col = key
        if i != 0:
            raise IndexError("MiniFrame supports only row 0")
        return self._f._row[self._f._col_index[col]]



class MiniFrame:
    """
    Remplacement minimal de DataFrame pour les tests unitaires sans pandas.
    Fournit les attributs columns, loc et shape nécessaires à la compatibilité.
    """
    def __init__(self, row: List[Any], columns: List[str]):
        self._row = row
        self.columns = list(columns)
        self._col_index = {c: i for i, c in enumerate(self.columns)}
        self.loc = _MiniLoc(self)

    @property
    def shape(self) -> Tuple[int, int]:
        """Retourne la forme (1, nb_colonnes) comme un DataFrame à une ligne."""
        return (1, len(self.columns))


# --- Construction optimisée ---------------------------------------------------

def build_row(
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
) -> Tuple[List[Any], List[int]]:
    """
    Construit une ligne d'entrée pour le modèle à partir du payload utilisateur.
    Gère l'ordre des features, les valeurs manquantes et la normalisation des features catégorielles.

    Args:
        payload (dict): Données d'entrée utilisateur.
        kept_features (list): Liste des features à garder (ordre attendu par le modèle).
        cat_features (list): Liste des features catégorielles.

    Returns:
        tuple: (ligne d'entrée, indices des features catégorielles)
    """
    # row dans l'ordre kept_features
    row: List[Any] = []
    for f in kept_features:
        v = payload.get(f, None)
        row.append(np.nan if v is None else v)

    # indices catégoriels (CatBoost veut les indices dans l'ordre des colonnes)
    cat_set = set(cat_features)
    cat_idx = [i for i, f in enumerate(kept_features) if f in cat_set]

    # normalisation catégorielle: NaN/None -> "__MISSING__", sinon str
    for i in cat_idx:
        v = row[i]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[i] = "__MISSING__"
        else:
            row[i] = str(v)

    return row, cat_idx


def build_X(
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
) -> MiniFrame:
    """
    Construit un objet MiniFrame compatible DataFrame pour les tests unitaires.

    Args:
        payload (dict): Données d'entrée utilisateur.
        kept_features (list): Liste des features à garder.
        cat_features (list): Liste des features catégorielles.

    Returns:
        MiniFrame: Objet minimaliste simulant un DataFrame.
    """
    row, _ = build_row(payload, kept_features, cat_features)
    return MiniFrame(row=row, columns=kept_features)


def _extract_proba_class1(pred: Any) -> float:
    """
    Extrait la probabilité de la classe 1 à partir de la sortie du modèle.

    Args:
        pred: Sortie du modèle (array ou liste).

    Returns:
        float: Probabilité de la classe 1.
    """
    arr = np.asarray(pred)
    if arr.size == 0:
        raise ValueError("Prediction output is empty")
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[0, 1])
    return float(arr.reshape(-1)[0])


def _call_model(fn, X, *, thread_count: int | None):
    """
    Appelle la fonction de prédiction du modèle, en passant thread_count si disponible.
    """
    if thread_count is None:
        return fn(X)
    try:
        sig = inspect.signature(fn)
        if "thread_count" in sig.parameters:
            return fn(X, thread_count=thread_count)
    except Exception:
        pass
    return fn(X)


def predict_score(
    model: Any,
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
    threshold: float,
    *,
    thread_count: int | None = None,
) -> Dict[str, Any]:
    """
    Calcule la prédiction (score et décision) à partir du modèle et des données utilisateur.
    Gère la compatibilité avec différents formats d'entrée pour le modèle.

    Args:
        model: Modèle CatBoost ou compatible.
        payload (dict): Données d'entrée utilisateur.
        kept_features (list): Liste des features à garder.
        cat_features (list): Liste des features catégorielles.
        threshold (float): Seuil de décision.
        thread_count (int, optionnel): Nombre de threads pour la prédiction.

    Returns:
        dict: Résultat de la prédiction (proba, score, décision, etc).
    
    example:
        >>> from app.model.predict import predict_score 
        >>> model = ...  # Charger un modèle CatBoost
        >>> payload = {"feature1": 10, "feature2": "A", ...}
        >>> kept_features = ["feature1", "feature2", ...]   
        >>> cat_features = ["feature2", ...]
        >>> threshold = 0.5
        >>> result = predict_score(model, payload, kept_features, cat_features, threshold)
        >>> print(result)
    """
    row, _ = build_row(payload, kept_features, cat_features)

    # 1) format optimisé pour CatBoost / prod
    X_fast = [row]  # liste 2D

    # 2) format "DataFrame-like" minimal pour compat tests
    X_df_like = MiniFrame(row=row, columns=kept_features)

    if hasattr(model, "predict_proba"):
        fn = model.predict_proba
    else:
        fn = model.predict

    # On tente d'abord le format optimisé, puis fallback si le modèle/test l'exige
    try:
        pred = _call_model(fn, X_fast, thread_count=thread_count)
    except Exception as e1:
        try:
            pred = _call_model(fn, X_df_like, thread_count=thread_count)
        except Exception:
            raise e1

    proba = _extract_proba_class1(pred)

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