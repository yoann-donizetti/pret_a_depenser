from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from core.config import PROJECT_ROOT, DATABASE_URL


MIGRATIONS_DIR = PROJECT_ROOT / "core" / "db" / "migrations"


def run_migration(conn: psycopg.Connection) -> None:
    sql_path = MIGRATIONS_DIR / "002_init_features_store.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"Migration introuvable: {sql_path}")
    conn.execute(sql_path.read_text(encoding="utf-8"))


def to_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        else:
            out[k] = v.item() if hasattr(v, "item") else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV API-ready (ex: data/processed/X_api.csv)")
    ap.add_argument("--chunksize", type=int, default=2000)
    args = ap.parse_args()

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL manquante (core.config).")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    upsert_sql = """
    INSERT INTO features_store (sk_id_curr, data)
    VALUES (%s, %s::jsonb)
    ON CONFLICT (sk_id_curr) DO UPDATE SET
      data = EXCLUDED.data,
      updated_at = now();
    """

    inserted = 0
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        run_migration(conn)

        for chunk in pd.read_csv(csv_path, chunksize=args.chunksize):
            if "SK_ID_CURR" not in chunk.columns:
                raise ValueError("Le CSV doit contenir la colonne SK_ID_CURR.")

            rows = []
            for _, r in chunk.iterrows():
                sk_id = int(r["SK_ID_CURR"])
                payload = to_payload(r.to_dict())
                rows.append((sk_id, Jsonb(payload)))

            with conn.cursor() as cur:
                cur.executemany(upsert_sql, rows)

            inserted += len(rows)

    print(f"OK: {inserted} lignes upsert dans features_store.")


if __name__ == "__main__":
    main()