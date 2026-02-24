from __future__ import annotations

from typing import Any, Dict, List, Set
import math

from app.utils.errors import ApiError


BINARY_FIELDS = {
    "REG_CITY_NOT_LIVE_CITY",
    "FLAG_DOCUMENT_3",
}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _finite_float(x: Any) -> float:
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
    kept_set: Set[str] = set(kept_features)
    cat_set: Set[str] = set(cat_features)
    num_set: Set[str] = (kept_set - cat_set) - BINARY_FIELDS

    # 0) Champs inconnus
    if reject_unknown_fields:
        unknown = [k for k in payload.keys() if k not in kept_set and k != "SK_ID_CURR"]
        if unknown:
            raise ApiError(
                code="UNKNOWN_FIELDS",
                message="Le payload contient des champs non attendus.",
                details={"unknown_fields": unknown[:30], "count": len(unknown)},
            )

    # 0bis) Champs manquants (Option A)
    missing = [f for f in kept_features if f not in payload]
    if missing:
        raise ApiError(
            code="MISSING_FIELDS",
            message="Le payload ne contient pas toutes les features attendues.",
            details={"missing_fields": missing[:30], "count": len(missing)},
        )

    # 1) SK_ID_CURR
    if "SK_ID_CURR" in payload and payload["SK_ID_CURR"] is not None:
        v = payload["SK_ID_CURR"]
        if not _is_int(v) or v <= 0:
            raise ApiError(
                code="INVALID_SK_ID_CURR",
                message="SK_ID_CURR doit être un entier > 0.",
                details={"SK_ID_CURR": v},
            )

    # 2) Binaires
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

    # Numériques
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

    # Catégorielles
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

    # 3) Bornes génériques
    for k in kept_set.intersection(payload.keys()):
        v = payload.get(k, None)
        if v is None:
            continue
        if k in BINARY_FIELDS:
            continue

        if k in num_set:
            f = float(v)

            if k.startswith("EXT_SOURCE_") and not (0.0 <= f <= 1.0):
                raise ApiError("OUT_OF_RANGE", f"{k} doit être dans [0, 1].", {"field": k, "value": f})

            if k.endswith("_MODE") and not (0.0 <= f <= 1.0):
                raise ApiError("OUT_OF_RANGE", f"{k} doit être dans [0, 1].", {"field": k, "value": f})

            if "UTILIZATION" in k and not (-0.01 <= f <= 3.0):
                raise ApiError("OUT_OF_RANGE", f"{k} incohérent.", {"field": k, "value": f})

            if "RATIO" in k and f < 0:
                raise ApiError("OUT_OF_RANGE", f"{k} doit être >= 0.", {"field": k, "value": f})

            if k.startswith("AMT_"):
                if k == "AMT_INCOME_TOTAL":
                    if f <= 0:
                        raise ApiError("OUT_OF_RANGE", f"{k} doit être > 0.", {"field": k, "value": f})
                else:
                    if f < 0:
                        raise ApiError("OUT_OF_RANGE", f"{k} doit être >= 0.", {"field": k, "value": f})

            if "CNT_" in k or k.endswith("_CNT") or k.endswith("_COUNT"):
                if f < 0:
                    raise ApiError("OUT_OF_RANGE", f"{k} doit être >= 0.", {"field": k, "value": f})

            if k.startswith("DAYS_"):
                if not (-100000 <= f <= 100000):
                    raise ApiError("OUT_OF_RANGE", f"{k} incohérent (borne).", {"field": k, "value": f})

                if k == "DAYS_BIRTH":
                    if f >= 0:
                        raise ApiError("OUT_OF_RANGE", "DAYS_BIRTH devrait être négatif.", {"field": k, "value": f})
                    age_years = abs(f) / 365.25
                    if not (0 < age_years < 120):
                        raise ApiError(
                            "OUT_OF_RANGE",
                            "Âge incohérent (DAYS_BIRTH).",
                            {"field": k, "value": f, "age_years": round(age_years, 2)},
                        )

            if k == "OWN_CAR_AGE" and (f < 0 or f > 100):
                raise ApiError("OUT_OF_RANGE", f"{k} incohérent.", {"field": k, "value": f})

    return payload