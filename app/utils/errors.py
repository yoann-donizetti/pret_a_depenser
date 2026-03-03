
"""
Définition d'une exception personnalisée pour l'API, permettant de structurer les erreurs retournées au client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ApiError(Exception):
    """
    Exception personnalisée pour les erreurs API.
    Permet de retourner un code, un message, des détails optionnels et un statut HTTP.
    """
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    http_status: int = 400

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'erreur en dictionnaire pour une réponse JSON.
        """
        out: Dict[str, Any] = {"error": self.code, "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out