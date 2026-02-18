from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_txt_list(path: Path) -> List[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("JSON vide.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Le JSON doit être un objet { ... } (1 seul client).")
    return data