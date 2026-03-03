"""
Script de génération et d'insertion des distributions de référence (ref_feature_dist) en base à partir d'un CSV.
Pour chaque feature, calcule la distribution de référence (numérique ou catégorielle) et l'insère/maj en base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
load_dotenv()

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from core.config import PROJECT_ROOT, DATABASE_URL


MIGRATIONS_DIR = PROJECT_ROOT / "core" / "db" / "migrations"
SQL_DIR = PROJECT_ROOT / "core" / "db" / "sql"
EXCLUDED_FEATURES = {"SK_ID_CURR"}

def load_sql(path: Path) -> str:
    """
    Charge le contenu d'un fichier SQL depuis le disque.
    Lève une FileNotFoundError si le fichier n'existe pas.
    """
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def infer_kind(s: pd.Series) -> str:
    """
    Détermine le type d'une feature (numérique ou catégorielle) à partir d'une série pandas.
    Retourne 'numeric' ou 'categorical'.
    """
    if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower() == "boolean":
        return "numeric"
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        return "categorical"
    return "numeric"


def numeric_ref_dist(s: pd.Series, bins: int):
    """
    Calcule la distribution de référence pour une feature numérique.
    Retourne : (bins_json, dist_json, n_ref)
    """
    if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower() == "boolean":
        s = s.astype("Int64")

    x = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(x))

    if n == 0:
        return {"edges": [0.0, 0.0]}, {"labels": ["__EMPTY__"], "p": [1.0]}, 0

    if x.nunique() < 2:
        v = float(x.iloc[0])
        return {"edges": [v, v]}, {"labels": [f"[{v},{v}]"], "p": [1.0]}, n

    qs = np.linspace(0, 1, bins + 1)

    try:
        edges = np.unique(x.quantile(qs).to_numpy(dtype=float))
    except Exception:
        edges = np.unique(np.quantile(x.to_numpy(dtype=float), qs))

    edges = np.unique(np.sort(edges))

    if len(edges) < 3:
        mn, mx = float(x.min()), float(x.max())
        if mn == mx:
            return {"edges": [mn, mx]}, {"labels": [f"[{mn},{mx}]"], "p": [1.0]}, n
        edges = np.array([mn, (mn + mx) / 2, mx], dtype=float)

    binned = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
    dist = binned.value_counts(normalize=True).sort_index()

    labels = [str(idx) for idx in dist.index]
    p = [float(v) for v in dist.values]

    if len(p) == 0:
        return {"edges": [float(edges[0]), float(edges[-1])]}, {"labels": ["__EMPTY__"], "p": [1.0]}, n

    return {"edges": [float(e) for e in edges]}, {"labels": labels, "p": p}, n


def categorical_ref_dist(s: pd.Series, topk: int):
    """
    Calcule la distribution de référence pour une feature catégorielle.
    Retourne : (bins_json, dist_json, n_ref)
    """
    x = s.fillna("__MISSING__").astype(str)
    n = int(len(x))

    vc = x.value_counts(normalize=True)
    top = vc.head(topk)
    other_p = float(max(0.0, 1.0 - top.sum()))

    labels = list(top.index.astype(str))
    p = [float(v) for v in top.values]

    if other_p > 0:
        labels.append("__OTHER__")
        p.append(other_p)

    return {"topk": topk}, {"labels": labels, "p": p}, n


def main() -> None:
    """
    Point d'entrée principal du script :
    - Charge un CSV de données de référence
    - Calcule la distribution de chaque feature (numérique/catégorielle)
    - Insère ou met à jour les distributions en base
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL manquante (core.config).")

    # 1) Chargement du CSV de données de référence
    df = pd.read_csv(Path(args.csv))

    # 2) Chargement des requêtes SQL nécessaires
    migration_sql = load_sql(MIGRATIONS_DIR / "003_init_ref_feature_dist.sql")
    upsert_sql = load_sql(SQL_DIR / "ref_feature_dist_upsert.sql")

    # 3) Calcul des distributions de référence pour chaque feature
    rows = []
    for col in df.columns:
        if col in EXCLUDED_FEATURES:
            continue
        kind = infer_kind(df[col])
        if kind == "numeric":
            bins_json, dist_json, n_ref = numeric_ref_dist(df[col], args.bins)
        else:
            bins_json, dist_json, n_ref = categorical_ref_dist(df[col], args.topk)

        rows.append(
            {
                "feature": col,
                "kind": kind,
                "bins_json": json.dumps(bins_json),
                "ref_dist_json": json.dumps(dist_json),
                "n_ref": n_ref,
            }
        )

    # 4) Insertion ou mise à jour des distributions en base
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(migration_sql)
        with conn.cursor() as cur:
            cur.executemany(upsert_sql, rows)

    # 5) Affichage du résultat
    print(f"OK: {len(rows)} features insérées/mises à jour.")


if __name__ == "__main__":
    main()