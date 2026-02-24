from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.utils.errors import ApiError


def load_txt_list(path: Path) -> List[str]:
    return [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def parse_json(text: str) -> Dict[str, Any]:
    if text is None or not str(text).strip():
        raise ApiError(code="EMPTY_JSON", message="Le JSON d'entrée est vide.")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ApiError(
            code="INVALID_JSON",
            message="JSON invalide.",
            details={"line": e.lineno, "col": e.colno, "msg": e.msg},
        )

    if not isinstance(obj, dict):
        raise ApiError(
            code="INVALID_PAYLOAD",
            message="Le JSON doit être un objet (dictionnaire).",
        )

    return obj