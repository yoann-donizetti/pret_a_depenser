# app/utils/prod_logging_db.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import psycopg

_CONN: Optional[psycopg.Connection] = None


def _get_conn() -> Optional[psycopg.Connection]:
    global _CONN

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    if _CONN is None or _CONN.closed:
        _CONN = psycopg.connect(db_url, autocommit=True)

    return _CONN


def init_db() -> None:
    conn = _get_conn()
    if conn is None:
        return  # pas de DB => pas de init

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
            outputs JSONB
        );
        """
    )


def log_event_db(event: Dict[str, Any]) -> None:
    conn = _get_conn()
    if conn is None:
        return  # pas de DB => pas de log

    inputs_json = json.dumps(event.get("inputs")) if event.get("inputs") is not None else None
    outputs_json = json.dumps(event.get("outputs")) if event.get("outputs") is not None else None

    conn.execute(
        """
        INSERT INTO prod_requests (endpoint, status_code, latency_ms, sk_id_curr, inputs, outputs)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
        """,
        (
            event.get("endpoint"),
            event.get("status_code"),
            event.get("latency_ms"),
            None if event.get("sk_id_curr") is None else str(event.get("sk_id_curr")),
            inputs_json,
            outputs_json,
        ),
    )