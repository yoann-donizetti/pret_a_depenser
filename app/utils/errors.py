# app/utils/errors.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ApiError(Exception):
    """
Cette classe définit une exception personnalisée (ApiError) utilisée pour gérer
les erreurs de validation et les erreurs métier dans l'API.

elle permet de standardiser le format des réponses d'erreur et de retourner
un code HTTP cohérent (400, 422, etc.) avec un message explicite et des détails.
"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    http_status: int = 400

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "error": self.code,
            "message": self.message,
        }
        if self.details is not None:
            out["details"] = self.details
        return out