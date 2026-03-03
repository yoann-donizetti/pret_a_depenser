
"""
Configuration centrale du projet :
 - Gestion des variables d'environnement et des chemins principaux
 - Centralisation des paramètres globaux (environnement, base, HuggingFace...)
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str | None = None) -> str | None:
    """
    Récupère la valeur d'une variable d'environnement, avec fallback et nettoyage.

    Args:
        key (str): Nom de la variable d'environnement.
        default (str | None): Valeur par défaut si non trouvée.

    Returns:
        str | None: Valeur nettoyée ou None si absente.
    """
    v = os.getenv(key, default)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV = (_env("ENV", "dev") or "dev").lower()
BUNDLE_SOURCE = (_env("BUNDLE_SOURCE", "auto") or "auto").lower()

DATABASE_URL = _env("DATABASE_URL")

HF_REPO_ID = _env("HF_REPO_ID")
HF_TOKEN = _env("HF_TOKEN")