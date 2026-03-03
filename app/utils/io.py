
"""
Fonctions utilitaires pour la lecture de listes texte et le parsing sécurisé de JSON avec gestion d'erreur API.
fonctions principales :
    - load_txt_list: Charge une liste de chaînes à partir d'un fichier texte (une valeur par ligne).
    - parse_json: Parse une chaîne JSON en dictionnaire Python, avec gestion d'erreur API explicite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.utils.errors import ApiError


def load_txt_list(path: Path) -> List[str]:
    """
    Charge une liste de chaînes à partir d'un fichier texte (une valeur par ligne).

    Args:
        path (Path): Chemin du fichier texte.

    Returns:
        List[str]: Liste des lignes non vides, sans espaces superflus.
    example:
        >>> from pathlib import Path
        >>> path = Path("fichier.txt")
        >>> lignes = load_txt_list(path)
    """
    return [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def parse_json(text: str) -> Dict[str, Any]:
    """
    Parse une chaîne JSON en dictionnaire Python, avec gestion d'erreur API explicite.

    Args:
        text (str): Chaîne JSON à parser.

    Returns:
        Dict[str, Any]: Dictionnaire issu du JSON.

    Raises:
        ApiError: Si le JSON est vide, invalide ou n'est pas un objet.
    """
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