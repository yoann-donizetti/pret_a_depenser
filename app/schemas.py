# app/schemas.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict

# On charge l'exemple une fois (au démarrage)
_EXAMPLE_PATH = Path("examples/input_example.json")

try:
    _EXAMPLE: Dict[str, Any] = json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    _EXAMPLE.pop("TARGET", None)
except Exception:
    # Fallback si le fichier n'est pas dispo (ex: tests/CI)
    _EXAMPLE = {"SK_ID_CURR": 100001}


class PredictRequest(BaseModel):
    """
    Option A: payload large (125 features).
    Pydantic accepte les champs extra.
    """
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": _EXAMPLE
        },
    )

    SK_ID_CURR: int


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]