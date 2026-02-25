from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from typing import Optional

import psycopg

_CONN: Optional[psycopg.Connection] = None


def get_conn() -> Optional[psycopg.Connection]:
    global _CONN
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    if _CONN is None or _CONN.closed:
        _CONN = psycopg.connect(db_url, autocommit=True)

    return _CONN


def _apply_migrations(conn: psycopg.Connection) -> None:
    mig_dir = Path(__file__).resolve().parent / "migrations"
    files = sorted(mig_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No migrations found in: {mig_dir}")

    for f in files:
        sql = f.read_text(encoding="utf-8")
        conn.execute(sql)


def init_db() -> None:
    """
    Applique les migrations SQL (idempotent).
    Si DATABASE_URL absent => no-op (API reste UP).
    """
    conn = get_conn()
    if conn is None:
        return

    _apply_migrations(conn)