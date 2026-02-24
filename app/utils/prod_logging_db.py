# app/utils/prod_logging_db.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import psycopg

_CONN: Optional[psycopg.Connection] = None


def _get_conn() -> Optional[psycopg.Connection]:
    """
    Retourne une connexion psycopg réutilisable.
    Si DATABASE_URL n'est pas défini, on désactive silencieusement la DB (l'API doit rester UP).
    """
    global _CONN
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    if _CONN is None or _CONN.closed:
        # autocommit=True pour exécuter CREATE/ALTER sans gérer explicitement les transactions
        _CONN = psycopg.connect(db_url, autocommit=True)

    return _CONN


def init_db() -> None:
    """
    Crée la table prod_requests si elle n'existe pas,
    puis applique des migrations "safe" (ADD COLUMN IF NOT EXISTS)
    pour garantir que le schéma est toujours à jour.
    """
    conn = _get_conn()
    if conn is None:
        return  # pas de DB => pas de init

    # 1) création minimale (si table absente)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prod_requests (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL DEFAULT now(),
            endpoint TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            latency_ms DOUBLE PRECISION,
            sk_id_curr TEXT,
            inputs JSONB,
            outputs JSONB,
            error TEXT,
            message TEXT
        );
        """
    )

    # 2) "migrations" idempotentes si la table existe déjà mais sans certaines colonnes
    # (utile sur Supabase si tu as déjà créé la table plus tôt avec moins de colonnes)
    conn.execute("ALTER TABLE prod_requests ADD COLUMN IF NOT EXISTS error TEXT;")
    conn.execute("ALTER TABLE prod_requests ADD COLUMN IF NOT EXISTS message TEXT;")

    # Optionnel: index utiles pour monitoring (safe aussi)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prod_requests_ts ON prod_requests(ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prod_requests_endpoint ON prod_requests(endpoint);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prod_requests_status_code ON prod_requests(status_code);")


def log_event_db(event: Dict[str, Any]) -> None:
    """
    Insère un event dans prod_requests.
    Le logging ne doit jamais casser l'app -> on laisse le try/except au niveau appelant (main._safe_log).
    """
    conn = _get_conn()
    if conn is None:
        return

    conn.execute(
        """
        INSERT INTO prod_requests (endpoint, status_code, latency_ms, sk_id_curr, inputs, outputs, error, message)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s);
        """,
        (
            event.get("endpoint"),
            int(event.get("status_code") or 0),
            event.get("latency_ms"),
            None if event.get("sk_id_curr") is None else str(event.get("sk_id_curr")),
            psycopg.types.json.Jsonb(event.get("inputs") or {}),
            psycopg.types.json.Jsonb(event.get("outputs") or {}),
            event.get("error"),
            event.get("message"),
        ),
    )