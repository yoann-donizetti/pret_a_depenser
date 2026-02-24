from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()


def infer_kind(s: pd.Series) -> str:
    """
    Détermine si une feature est 'categorical' ou 'numeric'.

    Règles:
    - bool -> numeric (0/1)
    - object / string -> categorical
    - sinon -> numeric
    """
    if pd.api.types.is_bool_dtype(s):
        return "numeric"
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        return "categorical"
    return "numeric"


def numeric_ref_dist(s: pd.Series, bins: int) -> tuple[dict, dict, int]:
    """
    Construit une distribution de référence pour une variable numérique.
    Bins calculés sur quantiles.
    """
    # bool -> int pour éviter crash numpy/pandas sur quantile
    if pd.api.types.is_bool_dtype(s):
        s = s.astype("int64")

    x = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(x))

    if n == 0:
        return {"edges": [0.0, 0.0], "labels": ["[0,0]"]}, {"labels": ["[0,0]"], "p": [1.0]}, 0

    if x.nunique() < 2:
        v = float(x.iloc[0])
        return {"edges": [v, v], "labels": [f"[{v},{v}]"]}, {"labels": [f"[{v},{v}]"], "p": [1.0]}, n

    qs = np.linspace(0, 1, bins + 1)

    # Quantiles -> edges
    edges = np.unique(x.quantile(qs).values)

    # Si trop peu d'edges, fallback sur numpy.quantile
    if len(edges) < 3:
        edges = np.unique(np.quantile(x.values, qs))

    edges = np.unique(edges)

    # Toujours trop peu -> bins artificiels
    if len(edges) < 3:
        mn, mx = float(x.min()), float(x.max())
        if mn == mx:
            return {"edges": [mn, mx], "labels": [f"[{mn},{mx}]"]}, {"labels": [f"[{mn},{mx}]"], "p": [1.0]}, n
        edges = np.array([mn, (mn + mx) / 2, mx], dtype=float)

    # Assure edges triés et uniques
    edges = np.unique(np.sort(edges))

    # Binning robuste
    try:
        binned = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
        dist = binned.value_counts(normalize=True).sort_index()

        labels = [str(idx) for idx in dist.index]
        p = [float(v) for v in dist.values]

        # Si jamais tout s'effondre (rare)
        if len(p) == 0:
            raise ValueError("Empty dist after cut")

        # Recalcule edges depuis les intervalles si duplicates drop a réduit
        # (optionnel, mais ça rend cohérent)
        return {"edges": [float(e) for e in edges], "labels": labels}, {"labels": labels, "p": p}, n

    except Exception:
        # Fallback: histogram uniforme en bins
        hist, hist_edges = np.histogram(x.values, bins=bins)
        total = hist.sum()
        if total == 0:
            return {"edges": [float(hist_edges[0]), float(hist_edges[-1])], "labels": ["__EMPTY__"]}, {"labels": ["__EMPTY__"], "p": [1.0]}, n

        p = (hist / total).astype(float).tolist()
        labels = [
            f"[{float(hist_edges[i])}, {float(hist_edges[i+1])}]"
            for i in range(len(hist_edges) - 1)
        ]
        return {"edges": [float(e) for e in hist_edges], "labels": labels}, {"labels": labels, "p": p}, n


def categorical_ref_dist(s: pd.Series, topk: int) -> tuple[dict, dict, int]:
    x = s.fillna("__MISSING__").astype(str)
    n = int(len(x))

    if n == 0:
        return {"topk": topk}, {"labels": ["__EMPTY__"], "p": [1.0]}, 0

    vc = x.value_counts(normalize=True)

    top = vc.head(topk)
    other_p = float(max(0.0, 1.0 - top.sum()))

    labels = list(top.index.astype(str))
    p = [float(v) for v in top.values]

    if other_p > 0:
        labels.append("__OTHER__")
        p.append(other_p)

    return {"topk": topk}, {"labels": labels, "p": p}, n


def upsert_ref(
    conn: psycopg.Connection,
    feature: str,
    kind: str,
    bins_json: dict,
    ref_dist_json: dict,
    n_ref: int,
) -> None:
    conn.execute(
        """
        INSERT INTO ref_feature_dist(feature, kind, bins_json, ref_dist_json, n_ref)
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
        ON CONFLICT (feature) DO UPDATE SET
          kind = EXCLUDED.kind,
          bins_json = EXCLUDED.bins_json,
          ref_dist_json = EXCLUDED.ref_dist_json,
          n_ref = EXCLUDED.n_ref,
          created_at = now();
        """,
        (feature, kind, json.dumps(bins_json), json.dumps(ref_dist_json), n_ref),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV de référence (train split final)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL manquante dans l'environnement (.env non chargé ?)")

    df = pd.read_csv(Path(args.csv))

    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ref_feature_dist (
              feature TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK (kind IN ('numeric','categorical')),
              bins_json JSONB,
              ref_dist_json JSONB NOT NULL,
              n_ref BIGINT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )

        for col in df.columns:
            kind = infer_kind(df[col])

            if kind == "numeric":
                bins_json, dist_json, n_ref = numeric_ref_dist(df[col], bins=args.bins)
            else:
                bins_json, dist_json, n_ref = categorical_ref_dist(df[col], topk=args.topk)

            upsert_ref(conn, col, kind, bins_json, dist_json, n_ref)

    print(f"OK: référence enregistrée pour {df.shape[1]} features.")


if __name__ == "__main__":
    main()