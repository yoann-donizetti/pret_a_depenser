# app/utils/validation.py

from __future__ import annotations

from typing import Any, Dict, List, Set
import math

from app.utils.errors import ApiError

BINARY_FIELDS = {
    "REG_CITY_NOT_LIVE_CITY",
    "FLAG_DOCUMENT_3",
}

def _is_number(x: Any) -> bool:
    # bool est un int en Python -> on l'exclut
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)

def _finite_float(x: Any) -> float:
    """Convertit en float et refuse NaN/inf."""
    f = float(x)
    if not math.isfinite(f):
        raise ValueError("non-finite")
    return f

def _is_binary_value(v: Any) -> bool:
    return isinstance(v, bool) or (_is_int(v) and v in (0, 1))

def validate_payload(
    payload: Dict[str, Any],
    kept_features: List[str],
    cat_features: List[str],
    *,
    reject_unknown_fields: bool = True,
) -> Dict[str, Any]:
    """
    Validation "max" basée sur le schéma :

    - Refuse les champs inconnus (optionnel)
    - Numériques attendues: int|float (bool refusé), NaN/inf refusés
    - Catégorielles attendues: str (sauf champs binaires gérés via BINARY_FIELDS)
    - Règles génériques de bornes (data-driven par famille)
    - None autorisés (le modèle gère les manquants)
    """

    kept_set: Set[str] = set(kept_features)
    cat_set: Set[str] = set(cat_features)

    # IMPORTANT: on retire les champs binaires du set numérique
    num_set: Set[str] = (kept_set - cat_set) - BINARY_FIELDS

    # -------------------------
    # 0) Champs inconnus
    # -------------------------
    if reject_unknown_fields:
        unknown = [k for k in payload.keys() if k not in kept_set and k != "SK_ID_CURR"]
        if unknown:
            raise ApiError(
                code="UNKNOWN_FIELDS",
                message="Le payload contient des champs non attendus.",
                details={"unknown_fields": unknown[:30], "count": len(unknown)},
            )

    # -------------------------
    # 1) SK_ID_CURR (si présent)
    # -------------------------
    if "SK_ID_CURR" in payload and payload["SK_ID_CURR"] is not None:
        v = payload["SK_ID_CURR"]
        if not _is_int(v) or v <= 0:
            raise ApiError(
                code="INVALID_SK_ID_CURR",
                message="SK_ID_CURR doit être un entier > 0.",
                details={"SK_ID_CURR": v},
            )

    # -------------------------
    # 2) Type checking strict (binaires d'abord)
    # -------------------------
    for k in BINARY_FIELDS:
        if k not in payload or payload[k] is None:
            continue
        v = payload[k]
        if not _is_binary_value(v):
            raise ApiError(
                code="INVALID_TYPE",
                message=f"Type invalide pour {k}. Attendu un bool ou 0/1.",
                details={"field": k, "value": v, "expected": "bool|0|1"},
            )

    # Numériques: doivent être number si non-None
    for k in num_set:
        if k not in payload or payload[k] is None:
            continue
        v = payload[k]
        if not _is_number(v):
            raise ApiError(
                code="INVALID_TYPE",
                message=f"Type invalide pour {k}. Attendu un nombre.",
                details={"field": k, "value": v, "expected": "number"},
            )
        try:
            _finite_float(v)
        except Exception:
            raise ApiError(
                code="INVALID_VALUE",
                message=f"Valeur invalide pour {k} (NaN/inf).",
                details={"field": k, "value": v},
            )

    # Catégorielles: str (les binaires ne passent pas ici)
    for k in cat_set:
        if k not in payload or payload[k] is None:
            continue
        if k in BINARY_FIELDS:
            continue
        v = payload[k]
        if not isinstance(v, str):
            raise ApiError(
                code="INVALID_TYPE",
                message=f"Type invalide pour {k}. Attendu une chaîne (str).",
                details={"field": k, "value": v, "expected": "str"},
            )

    # -------------------------
    # 3) Bornes génériques (par nom de feature)
    # -------------------------
    for k in kept_set.intersection(payload.keys()):
        v = payload.get(k, None)
        if v is None:
            continue

        # binaires déjà validés
        if k in BINARY_FIELDS:
            continue

        # numériques : bornes
        if k in num_set:
            f = float(v)

            # EXT_SOURCE_* : [0,1]
            if k.startswith("EXT_SOURCE_"):
                if not (0.0 <= f <= 1.0):
                    raise ApiError("OUT_OF_RANGE", f"{k} doit être dans [0, 1].", {"field": k, "value": f})

            # *_MODE : [0,1]
            if k.endswith("_MODE"):
                if not (0.0 <= f <= 1.0):
                    raise ApiError("OUT_OF_RANGE", f"{k} doit être dans [0, 1].", {"field": k, "value": f})

            # UTILIZATION : chez toi ça dépasse 1 et peut être très légèrement négatif
            if "UTILIZATION" in k:
                if not (-0.01 <= f <= 3.0):
                    raise ApiError("OUT_OF_RANGE", f"{k} incohérent.", {"field": k, "value": f})

            # RATIO : chez toi ça peut être >> 1, donc on impose juste >= 0
            if "RATIO" in k:
                if f < 0:
                    raise ApiError("OUT_OF_RANGE", f"{k} doit être >= 0.", {"field": k, "value": f})

            # Montants
            if k.startswith("AMT_"):
                if k == "AMT_INCOME_TOTAL":
                    if f <= 0:
                        raise ApiError("OUT_OF_RANGE", f"{k} doit être > 0.", {"field": k, "value": f})
                else:
                    if f < 0:
                        raise ApiError("OUT_OF_RANGE", f"{k} doit être >= 0.", {"field": k, "value": f})

            # Compteurs
            if "CNT_" in k or k.endswith("_CNT") or k.endswith("_COUNT"):
                if f < 0:
                    raise ApiError("OUT_OF_RANGE", f"{k} doit être >= 0.", {"field": k, "value": f})

            # Jours
            if k.startswith("DAYS_"):
                if not (-100000 <= f <= 100000):
                    raise ApiError("OUT_OF_RANGE", f"{k} incohérent (borne).", {"field": k, "value": f})

                if k == "DAYS_BIRTH":
                    if f >= 0:
                        raise ApiError("OUT_OF_RANGE", "DAYS_BIRTH devrait être négatif (âge en jours).", {"field": k, "value": f})
                    age_years = abs(f) / 365.25
                    if not (0 < age_years < 120):
                        raise ApiError(
                            "OUT_OF_RANGE",
                            "Âge incohérent (DAYS_BIRTH).",
                            {"field": k, "value": f, "age_years": round(age_years, 2)},
                        )

            # Âge voiture
            if k == "OWN_CAR_AGE":
                if f < 0 or f > 100:
                    raise ApiError("OUT_OF_RANGE", f"{k} incohérent.", {"field": k, "value": f})

    return payload