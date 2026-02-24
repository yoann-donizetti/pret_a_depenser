from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ApiError(Exception):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    http_status: int = 400

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"error": self.code, "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out