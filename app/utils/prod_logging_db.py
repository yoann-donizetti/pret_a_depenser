# app/utils/prod_logging_db.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import psycopg
from psycopg.types.json import Jsonb

_CONN: Optional[psycopg.Connection] = None


def _get_conn() -> psycopg.Connection:
    global _CONN
    if _CONN is None or _CONN.closed:
        db_url = os.environ["DATABASE_URL"]  # doit exister
        _CONN = psycopg.connect(db_url, autocommit=True)
    return _CONN


def init_db() -> None:
    conn = _get_conn()
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


def log_event_db(event: Dict[str, Any]) -> None:
    conn = _get_conn()

    status_code = int(event.get("status_code", 0))

    inputs = event.get("inputs")
    outputs = event.get("outputs")

    try:
        conn.execute(
            """
            INSERT INTO prod_requests (endpoint, status_code, latency_ms, sk_id_curr, inputs, outputs, error, message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                event.get("endpoint"),
                status_code,
                event.get("latency_ms"),
                None if event.get("sk_id_curr") is None else str(event.get("sk_id_curr")),
                Jsonb(inputs) if inputs is not None else None,
                Jsonb(outputs) if outputs is not None else None,
                event.get("error"),
                event.get("message"),
            ),
        )
    except Exception as e:
        # DEBUG (temporaire)
        print(f"[prod_logging_db] INSERT FAILED: {e}")
        print(f"[prod_logging_db] event keys: {list(event.keys())}")
        raise