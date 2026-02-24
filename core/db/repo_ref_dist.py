from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from psycopg.types.json import Jsonb

from core.db.conn import get_conn

_SQL_DIR = Path(__file__).resolve().parent / "sql"
_UPSERT_SQL = (_SQL_DIR / "ref_feature_dist_upsert.sql").read_text(encoding="utf-8")
_SELECT_ALL_SQL = (_SQL_DIR / "ref_feature_dist_select_all.sql").read_text(encoding="utf-8")
_SELECT_ONE_SQL = (_SQL_DIR / "ref_feature_dist_select_one.sql").read_text(encoding="utf-8")


def upsert_ref_feature_dist(
    *, feature: str, kind: str, bins_json: Dict[str, Any] | None, ref_dist_json: Dict[str, Any], n_ref: int
) -> None:
    conn = get_conn()
    if conn is None:
        return

    conn.execute(
        _UPSERT_SQL,
        {
            "feature": feature,
            "kind": kind,
            "bins_json": Jsonb(bins_json) if bins_json is not None else None,
            "ref_dist_json": Jsonb(ref_dist_json),
            "n_ref": int(n_ref),
        },
    )


def load_all_ref() -> List[Dict[str, Any]]:
    conn = get_conn()
    if conn is None:
        return []

    rows = conn.execute(_SELECT_ALL_SQL).fetchall()
    out: List[Dict[str, Any]] = []
    for (feature, kind, bins_json, ref_dist_json, n_ref, created_at) in rows:
        out.append(
            {
                "feature": feature,
                "kind": kind,
                "bins_json": bins_json,
                "ref_dist_json": ref_dist_json,
                "n_ref": n_ref,
                "created_at": created_at,
            }
        )
    return out


def load_one_ref(feature: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    if conn is None:
        return None

    row = conn.execute(_SELECT_ONE_SQL, {"feature": feature}).fetchone()
    if not row:
        return None

    (feature, kind, bins_json, ref_dist_json, n_ref, created_at) = row
    return {
        "feature": feature,
        "kind": kind,
        "bins_json": bins_json,
        "ref_dist_json": ref_dist_json,
        "n_ref": n_ref,
        "created_at": created_at,
    }