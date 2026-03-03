from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from psycopg.types.json import Jsonb

from core.db.conn import get_conn

_SQL_DIR = Path(__file__).resolve().parent / "sql"
_SELECT_SQL = (_SQL_DIR / "features_store_select_by_id.sql").read_text(encoding="utf-8")
_UPSERT_SQL = (_SQL_DIR / "features_store_upsert.sql").read_text(encoding="utf-8")


def get_features_by_id(sk_id_curr: int) -> Optional[Dict[str, Any]]:
    """
    Récupère les features associées à un identifiant client (sk_id_curr) depuis la base de données.
    
    Paramètres :
        sk_id_curr (int) : Identifiant du client.
    
    Retour :
        dict ou None : Dictionnaire des features si trouvé, sinon None.
    """
    conn = get_conn()
    if conn is None:
        return None

    row = conn.execute(_SELECT_SQL, {"sk_id_curr": int(sk_id_curr)}).fetchone()
    if not row:
        return None

    return row[0]  # JSONB -> dict


def upsert_features(sk_id_curr: int, data: Dict[str, Any]) -> None:
    """
    Insère ou met à jour les features d'un client dans la base de données.
    
    Paramètres :
        sk_id_curr (int) : Identifiant du client.
        data (dict) : Dictionnaire des features à insérer ou mettre à jour.
    """
    conn = get_conn()
    if conn is None:
        return

    conn.execute(_UPSERT_SQL, {"sk_id_curr": int(sk_id_curr), "data": Jsonb(data)})