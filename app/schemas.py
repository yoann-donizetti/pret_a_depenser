"""
Schémas Pydantic pour la validation et la documentation des requêtes/réponses de l'API.
 - Décrit les formats attendus pour la prédiction et le healthcheck
 - Fournit des exemples pour la documentation Swagger
"""
# app/schemas.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import StrictInt

# Exemple affiché dans /docs (Swagger)
_EXAMPLE_PATH = Path("examples/input_example.json")

try:
    _raw: Dict[str, Any] = json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    _EXAMPLE = {"SK_ID_CURR": int(_raw.get("SK_ID_CURR", 100001))}
except Exception:
    _EXAMPLE = {"SK_ID_CURR": 100001}


class PredictRequest(BaseModel):
    """
    Requête de prédiction :
    L'API reçoit uniquement SK_ID_CURR, puis va chercher les features en base.
    """
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE})

    SK_ID_CURR: StrictInt = Field(..., gt=0)


class PredictResponse(BaseModel):
    """
    Réponse de l'API pour une prédiction.
    Contient la probabilité, le score, la décision, le seuil et la latence.
    """
    SK_ID_CURR: Optional[int] = None
    proba_default: float
    score: int
    decision: Literal["ACCEPTED", "REFUSED"]
    threshold: float
    latency_ms: float


class HealthResponse(BaseModel):
    """
    Réponse pour l'endpoint de healthcheck.
    """
    status: Literal["ok", "not_ready"]