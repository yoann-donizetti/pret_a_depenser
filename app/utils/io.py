# app/utils/io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.utils.errors import ApiError


def load_txt_list(path: Path) -> List[str]:
    """
    Charge un fichier texte contenant une liste d'éléments (1 élément par ligne).

    Fonctionnement :
    - lit le fichier en UTF-8
    - supprime les espaces en début/fin de ligne
    - ignore les lignes vides
    - retourne une liste de strings propres
    """
    return [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def parse_json(text: str) -> Dict[str, Any]:
    """
    Parse un texte JSON reçu en entrée (souvent via Gradio / API).

    Objectif :
    - détecter les JSON vides
    - détecter les JSON invalides (erreur de syntaxe)
    - garantir que le JSON est un dictionnaire (objet) et pas une liste ou autre

    En cas de problème :
    - lève une ApiError avec un code clair, utilisable dans les réponses API
    """

    # Vérifie que le texte n'est pas vide ou composé uniquement d'espaces
    if text is None or not str(text).strip():
        raise ApiError(code="EMPTY_JSON", message="Le JSON d'entrée est vide.")

    # Tente de parser le JSON
    try:
        obj = json.loads(text)

    # Si la syntaxe JSON est invalide, on capture l'erreur
    # et on renvoie une erreur structurée avec ligne/colonne
    except json.JSONDecodeError as e:
        raise ApiError(
            code="INVALID_JSON",
            message="JSON invalide.",
            details={"line": e.lineno, "col": e.colno, "msg": e.msg},
        )

    # Vérifie que le JSON est bien un objet { ... } (dict)
    # et pas une liste [ ... ] ou autre type
    if not isinstance(obj, dict):
        raise ApiError(
            code="INVALID_PAYLOAD",
            message="Le JSON doit être un objet (dictionnaire)."
        )

    # Si tout est OK, retourne le dictionnaire Python
    return obj