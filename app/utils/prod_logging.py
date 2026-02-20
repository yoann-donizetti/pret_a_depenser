# app/utils/prod_logging.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path(os.getenv("PROD_LOG_DIR", "prod_logs"))
LOG_PATH = LOG_DIR / "requests.jsonl"


def log_event(event: Dict[str, Any]) -> None:
    """
    Enregistre un événement unique en format JSON Lines (JSONL) dans un fichier journal.
    
    Cette fonction est destinée à la surveillance en production et permet de suivre
    les entrées, les sorties, la latence et les erreurs. Chaque événement est écrit
    sous forme d'une seule ligne JSON pour faciliter le traitement des flux de journaux.
    
    Args:
        event (Dict[str, Any]): Un dictionnaire contenant les données de l'événement
            à enregistrer. Le dictionnaire peut inclure n'importe quelle paire clé-valeur
            relevant pour la surveillance.
    
    Returns:
        None
    
    Notes:
        - Si l'événement ne contient pas de clé "ts" (horodatage), une clé est
          automatiquement ajoutée avec l'heure UTC actuelle au format ISO 8601.
        - Le répertoire journal est créé automatiquement s'il n'existe pas.
        - Les événements sont ajoutés au fichier journal au format JSONL (un objet JSON par ligne).
        - Les caractères non-ASCII sont conservés dans la sortie.
    
    Exemples:
    >>> log_event({"message": "Connexion utilisateur", "user_id": 123})
    >>> log_event({"error": "Échec de la connexion à la base de données", "retry_count": 3})
    """
    # Crée le répertoire de journalisation s'il n'existe pas
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Crée une copie de l'événement et ajoute un horodatage s'il n'existe pas
    evt = dict(event)
    evt.setdefault("ts", datetime.now(timezone.utc).isoformat())

    # Écrit l'événement en format JSONL
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")