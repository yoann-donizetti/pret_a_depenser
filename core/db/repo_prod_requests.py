
# Module de gestion des requêtes de production :
# Permet d'insérer et de récupérer les requêtes faites à l'API en base de données pour le suivi et la traçabilité.
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from psycopg.types.json import Jsonb

from core.db.conn import get_conn

_SQL_DIR = Path(__file__).resolve().parent / "sql"
_INSERT_SQL = (_SQL_DIR / "prod_requests_insert.sql").read_text(encoding="utf-8")
_SELECT_SQL = (_SQL_DIR / "prod_requests_select.sql").read_text(encoding="utf-8")


def insert_prod_request(event: Dict[str, Any]) -> None:
    """
    Insère une requête de production dans la base de données.
    
    Paramètres :
        event (dict) : Dictionnaire contenant les informations de la requête (endpoint, status_code, latency_ms, sk_id_curr, inputs, outputs, error, message).
    """
    conn = get_conn()
    if conn is None:
        return

    params = {
        "endpoint": event.get("endpoint"),
        "status_code": int(event.get("status_code") or 0),
        "latency_ms": event.get("latency_ms"),
        "sk_id_curr": None if event.get("sk_id_curr") is None else str(event.get("sk_id_curr")),
        "inputs": Jsonb(event.get("inputs") or {}),
        "outputs": Jsonb(event.get("outputs") or {}),
        "error": event.get("error"),
        "message": event.get("message"),
    }
    conn.execute(_INSERT_SQL, params)


def select_prod_requests(endpoint: str = "/predict", limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Récupère les requêtes de production enregistrées pour un endpoint donné.
    
    Paramètres :
        endpoint (str) : Nom de l'endpoint à filtrer (par défaut '/predict').
        limit (int) : Nombre maximum de requêtes à retourner (par défaut 1000).
    
    Retour :
        Liste de dictionnaires contenant les informations des requêtes.
    """
    conn = get_conn()
    if conn is None:
        return []

    rows = conn.execute(_SELECT_SQL, {"endpoint": endpoint, "limit": int(limit)}).fetchall()

    out: List[Dict[str, Any]] = []
    for (ts, ep, status, latency, sk, inputs, outputs, error, message) in rows:
        out.append(
            {
                "ts": ts,
                "endpoint": ep,
                "status_code": status,
                "latency_ms": latency,
                "sk_id_curr": sk,
                "inputs": inputs or {},
                "outputs": outputs or {},
                "error": error,
                "message": message,
            }
        )

    out.reverse()  # chrono
    return out