# monitoring/streamlit_app.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg
import streamlit as st
import plotly.express as px

# Optionnel en local (si tu as un .env)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# -----------------------
# DB reads
# -----------------------
def read_ref_from_db(db_url: str) -> pd.DataFrame:
    """
    Lit la table ref_feature_dist générée par scripts/build_reference_dist.py

    Colonnes attendues:
    - feature (PK)
    - kind ('numeric'|'categorical')
    - bins_json (JSONB, optionnel)
    - ref_dist_json (JSONB)
    - n_ref
    """
    q = """
    SELECT feature, kind, bins_json, ref_dist_json, n_ref, created_at
    FROM ref_feature_dist
    ORDER BY feature
    """
    with psycopg.connect(db_url) as conn:
        rows = conn.execute(q).fetchall()

    return pd.DataFrame(
        rows,
        columns=["feature", "kind", "bins_json", "ref_dist_json", "n_ref", "created_at"],
    )


def read_prod_from_db(
    db_url: str,
    limit: int | None = None,
    endpoint: str = "/predict",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit les requêtes loggées dans prod_requests

    Retourne:
    - prod_X : DataFrame des inputs (colonnes = features)
    - prod_meta : DataFrame des métadonnées (ts, status_code, latency_ms, etc.)
    """
    q = """
    SELECT
      id,
      ts,
      endpoint,
      status_code,
      latency_ms,
      sk_id_curr,
      inputs,
      outputs,
      error
    FROM prod_requests
    WHERE endpoint = %s
    ORDER BY id DESC
    """
    params: list[Any] = [endpoint]
    if limit is not None:
        q += " LIMIT %s"
        params.append(limit)

    with psycopg.connect(db_url) as conn:
        rows = conn.execute(q, params).fetchall()

    meta_rows: List[Dict[str, Any]] = []
    inputs_rows: List[Dict[str, Any]] = []

    for (rid, ts, ep, status_code, latency_ms, sk_id_curr, inputs, outputs, error) in rows:
        meta_rows.append(
            {
                "id": rid,
                "ts": ts,
                "endpoint": ep,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "sk_id_curr": sk_id_curr,
                "error": error,
            }
        )
        # inputs est JSONB => psycopg renvoie souvent un dict Python
        inputs_rows.append(inputs or {})

    prod_meta = pd.DataFrame(meta_rows)
    prod_X = pd.DataFrame(inputs_rows)

    # Remettre ordre chrono (optionnel)
    prod_meta = prod_meta.iloc[::-1].reset_index(drop=True)
    prod_X = prod_X.iloc[::-1].reset_index(drop=True)

    return prod_X, prod_meta


# -----------------------
# PSI from distributions
# -----------------------
def psi_from_dist(ref_p: List[float], prod_p: List[float]) -> float:
    """
    PSI à partir de 2 distributions alignées (mêmes labels, même longueur)
    """
    eps = 1e-6
    r = np.clip(np.array(ref_p, dtype=float), eps, 1)
    p = np.clip(np.array(prod_p, dtype=float), eps, 1)
    return float(np.sum((p - r) * np.log(p / r)))


def prod_dist_numeric(prod_s: pd.Series, edges: List[float], labels_ref: List[str]) -> List[float]:
    """
    Construit la distribution prod alignée sur les bins de ref (edges+labels)
    - labels_ref doivent correspondre à str(pd.Interval) comme on a stocké côté ref
    """
    x = pd.to_numeric(prod_s, errors="coerce").dropna()
    if len(x) == 0:
        return [0.0] * len(labels_ref)

    # Binning avec les edges ref
    binned = pd.cut(x, bins=np.array(edges, dtype=float), include_lowest=True)
    dist = binned.value_counts(normalize=True).sort_index()

    dist_by_label = {str(k): float(v) for k, v in dist.items()}
    return [dist_by_label.get(lbl, 0.0) for lbl in labels_ref]


def prod_dist_categorical(prod_s: pd.Series, labels_ref: List[str]) -> List[float]:
    """
    Construit la distribution prod alignée sur labels_ref (topk + éventuellement __OTHER__)
    """
    x = prod_s.fillna("__MISSING__").astype(str)
    if len(x) == 0:
        return [0.0] * len(labels_ref)

    vc = x.value_counts(normalize=True)

    has_other = "__OTHER__" in labels_ref
    ref_cats = [l for l in labels_ref if l != "__OTHER__"]

    p = [float(vc.get(cat, 0.0)) for cat in ref_cats]

    if has_other:
        other = float(1.0 - sum(p))
        p.append(max(0.0, other))

    return p


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Prêt à Dépenser — Monitoring", layout="wide")
st.title("Monitoring — API & Data Drift (Option 1 : référence en DB)")


default_db_url = os.getenv("DATABASE_URL", "postgresql://pad:pad@127.0.0.1:5433/pad_monitoring")

with st.sidebar:
    st.header("Inputs")
    db_url = st.text_input("DATABASE_URL", default_db_url)
    endpoint = st.text_input("Endpoint", "/predict")
    limit = st.number_input("Nb requêtes à analyser (0 = tout)", min_value=0, value=1000, step=100)
    st.caption("Seuils PSI: <0.1 OK | 0.1–0.25 à surveiller | >0.25 drift fort")

limit_val = None if int(limit) == 0 else int(limit)

# ---- Load DB data
try:
    ref_df = read_ref_from_db(db_url=db_url)
except Exception as e:
    st.error(f"Erreur DB (ref_feature_dist): {e}")
    st.stop()

if len(ref_df) == 0:
    st.warning("Aucune référence trouvée en base (table ref_feature_dist vide). "
               "As-tu lancé scripts/build_reference_dist.py ?")
    st.stop()

try:
    prod_X, prod_meta = read_prod_from_db(db_url=db_url, limit=limit_val, endpoint=endpoint)
except Exception as e:
    st.error(f"Erreur DB (prod_requests): {e}")
    st.stop()

if len(prod_meta) == 0:
    st.warning("Aucune ligne trouvée en base (prod_requests). Fais tourner simulate_requests.py.")
    st.stop()

st.success(
    f"Référence DB: {len(ref_df)} features | "
    f"Prod: {len(prod_meta)} requêtes | prod inputs: {prod_X.shape[0]}×{prod_X.shape[1]}"
)

# -----------------------
# 1) OPS metrics
# -----------------------
st.subheader("1) Santé API (ops)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Nb requêtes", int(len(prod_meta)))
with c2:
    st.metric("Taux succès (200)", f"{(prod_meta['status_code'] == 200).mean() * 100:.2f}%")
with c3:
    med = pd.to_numeric(prod_meta["latency_ms"], errors="coerce").median()
    st.metric("Latence médiane (ms)", f"{float(med) if pd.notna(med) else np.nan:.2f}")
with c4:
    p95 = pd.to_numeric(prod_meta["latency_ms"], errors="coerce").quantile(0.95)
    st.metric("Latence p95 (ms)", f"{float(p95) if pd.notna(p95) else np.nan:.2f}")

lat = pd.to_numeric(prod_meta["latency_ms"], errors="coerce")
st.plotly_chart(px.histogram(lat.dropna(), nbins=30, title="Distribution latence (ms)"), use_container_width=True)

status_counts = prod_meta["status_code"].value_counts(dropna=False).sort_index()
st.plotly_chart(px.bar(status_counts, title="Codes HTTP"), use_container_width=True)

# -----------------------
# 2) Drift (PSI) using DB reference
# -----------------------
st.subheader("2) Data Drift (PSI) — Référence en DB")

# On calcule PSI uniquement pour les features présentes en ref
rows = []
missing_in_prod = []

for _, r in ref_df.iterrows():
    feat = r["feature"]
    kind = r["kind"]
    bins_json = r["bins_json"] or {}
    ref_dist_json = r["ref_dist_json"] or {}

    labels_ref = ref_dist_json.get("labels", [])
    ref_p = ref_dist_json.get("p", [])

    if feat not in prod_X.columns:
        missing_in_prod.append(feat)
        continue

    prod_s = prod_X[feat]

    if kind == "numeric":
        edges = (bins_json.get("edges") or [])
        # Si pas d'edges (référence dégénérée), on fait un PSI simple (souvent 0 / nan)
        if not edges or not labels_ref or not ref_p:
            v = np.nan
        else:
            prod_p = prod_dist_numeric(prod_s, edges=edges, labels_ref=labels_ref)
            v = psi_from_dist(ref_p, prod_p)
        rows.append({"feature": feat, "psi": v, "type": "numeric"})
    else:
        if not labels_ref or not ref_p:
            v = np.nan
        else:
            prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref)
            v = psi_from_dist(ref_p, prod_p)
        rows.append({"feature": feat, "psi": v, "type": "categorical"})

if missing_in_prod:
    st.warning(f"{len(missing_in_prod)} features de référence absentes en prod (ex: {missing_in_prod[:5]})")

drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
st.dataframe(drift_df, use_container_width=True, hide_index=True)

topk = drift_df.dropna().head(20)
st.plotly_chart(
    px.bar(topk[::-1], x="psi", y="feature", orientation="h", title="Top 20 PSI (drift)"),
    use_container_width=True,
)

# -----------------------
# 3) Inspect feature (ref dist vs prod dist)
# -----------------------
st.subheader("3) Détail feature (référence DB vs production)")

available_feats = drift_df["feature"].tolist()
feat = st.selectbox("Choisir une feature", available_feats)

ref_row = ref_df.loc[ref_df["feature"] == feat].iloc[0]
kind = ref_row["kind"]
bins_json = ref_row["bins_json"] or {}
ref_dist_json = ref_row["ref_dist_json"] or {}

labels_ref = ref_dist_json.get("labels", [])
ref_p = ref_dist_json.get("p", [])

prod_s = prod_X[feat]

if kind == "numeric":
    edges = bins_json.get("edges") or []
    prod_p = prod_dist_numeric(prod_s, edges=edges, labels_ref=labels_ref) if edges and labels_ref else []
    psi_val = psi_from_dist(ref_p, prod_p) if ref_p and prod_p else np.nan

    cA, cB = st.columns(2)
    with cA:
        st.write("Référence (bins + dist)")
        st.json({"edges": edges, "labels": labels_ref, "p": ref_p})
    with cB:
        st.write("Production (dist alignée)")
        st.json({"labels": labels_ref, "p": prod_p})

    st.metric("PSI", f"{psi_val:.6f}" if pd.notna(psi_val) else "nan")

    # Visualisation “bar” sur labels (plus fiable que overlay hist quand bins custom)
    plot_df = pd.DataFrame({"bin": labels_ref, "ref": ref_p, "prod": prod_p})
    st.plotly_chart(px.bar(plot_df, x="bin", y=["ref", "prod"], barmode="group", title=f"Dist par bins: {feat}"),
                    use_container_width=True)

else:
    prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref) if labels_ref else []
    psi_val = psi_from_dist(ref_p, prod_p) if ref_p and prod_p else np.nan

    cA, cB = st.columns(2)
    with cA:
        st.write("Référence (labels + dist)")
        st.json({"labels": labels_ref, "p": ref_p})
    with cB:
        st.write("Production (dist alignée)")
        st.json({"labels": labels_ref, "p": prod_p})

    st.metric("PSI", f"{psi_val:.6f}" if pd.notna(psi_val) else "nan")

    plot_df = pd.DataFrame({"cat": labels_ref, "ref": ref_p, "prod": prod_p})
    st.plotly_chart(px.bar(plot_df, x="cat", y=["ref", "prod"], barmode="group", title=f"Dist catégories: {feat}"),
                    use_container_width=True)