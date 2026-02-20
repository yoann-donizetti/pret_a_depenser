from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg
import streamlit as st
import plotly.express as px


# -----------------------
# DB read
# -----------------------

def read_prod_from_db(
    db_url: str,
    limit: int | None = None,
    endpoint: str = "/predict",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne:
    - prod_X : DataFrame des inputs (colonnes = features)
    - prod_meta : DataFrame des métadonnées (ts, status_code, latency_ms, etc.)
    """
    q = """
    SELECT
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
    params = [endpoint]
    if limit is not None:
        q += " LIMIT %s"
        params.append(limit)

    with psycopg.connect(db_url) as conn:
        rows = conn.execute(q, params).fetchall()

    # rows = list of tuples in same order as SELECT
    meta_rows: List[Dict[str, Any]] = []
    inputs_rows: List[Dict[str, Any]] = []

    for (ts, endpoint, status_code, latency_ms, sk_id_curr, inputs, outputs, error) in rows:
        meta_rows.append({
            "ts": ts,
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "sk_id_curr": sk_id_curr,
            "error": error,
        })
        inputs_rows.append(inputs or {})  # inputs est déjà un dict python (jsonb)

    prod_meta = pd.DataFrame(meta_rows)
    prod_X = pd.DataFrame(inputs_rows)

    # On remet l'ordre chronologique (optionnel mais souvent plus clair)
    prod_meta = prod_meta.iloc[::-1].reset_index(drop=True)
    prod_X = prod_X.iloc[::-1].reset_index(drop=True)

    return prod_X, prod_meta


# -----------------------
# PSI utils (inchangés)
# -----------------------

def psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    ref = ref.dropna()
    prod = prod.dropna()
    if len(ref) < 50 or len(prod) < 50:
        return np.nan
    if ref.nunique() < 2 or prod.nunique() < 2:
        return 0.0

    try:
        quantiles = np.linspace(0, 1, bins + 1)
        cuts = ref.quantile(quantiles).values
        cuts = np.unique(cuts)
        if len(cuts) < 3:
            return 0.0

        ref_bins = pd.cut(ref, bins=cuts, include_lowest=True)
        prod_bins = pd.cut(prod, bins=cuts, include_lowest=True)

        ref_dist = ref_bins.value_counts(normalize=True).sort_index()
        prod_dist = prod_bins.value_counts(normalize=True).sort_index()

        prod_dist = prod_dist.reindex(ref_dist.index).fillna(0)

        eps = 1e-6
        r = np.clip(ref_dist.values, eps, 1)
        p = np.clip(prod_dist.values, eps, 1)

        return float(np.sum((p - r) * np.log(p / r)))
    except Exception:
        return np.nan


def psi_categorical(ref: pd.Series, prod: pd.Series) -> float:
    ref = ref.fillna("__MISSING__").astype(str)
    prod = prod.fillna("__MISSING__").astype(str)

    ref_dist = ref.value_counts(normalize=True)
    prod_dist = prod.value_counts(normalize=True)

    idx = ref_dist.index.union(prod_dist.index)
    ref_dist = ref_dist.reindex(idx).fillna(0)
    prod_dist = prod_dist.reindex(idx).fillna(0)

    eps = 1e-6
    r = np.clip(ref_dist.values, eps, 1)
    p = np.clip(prod_dist.values, eps, 1)

    return float(np.sum((p - r) * np.log(p / r)))


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Prêt à Dépenser — Monitoring", layout="wide")
st.title("Monitoring — API & Data Drift (PoC local)")

default_ref = Path.cwd() / "data" / "processed" / "train_split_final.csv"
default_db_url = os.getenv("DATABASE_URL", "postgresql://pad:pad@127.0.0.1:5432/pad_monitoring")

with st.sidebar:
    st.header("Inputs")
    ref_path = st.text_input("Reference CSV", str(default_ref))
    db_url = st.text_input("DATABASE_URL", default_db_url)
    endpoint = st.text_input("Endpoint", "/predict")
    limit = st.number_input("Nb requêtes à analyser (0 = tout)", min_value=0, value=1000, step=100)
    bins = st.slider("PSI bins (numériques)", 5, 20, 10)
    st.caption("Seuils PSI: <0.1 OK | 0.1–0.25 à surveiller | >0.25 drift fort")

ref_path = Path(ref_path)
if not ref_path.exists():
    st.error(f"Reference introuvable: {ref_path}")
    st.stop()

ref_df = pd.read_csv(ref_path)
st.success(f"Reference chargée: {ref_df.shape[0]} lignes × {ref_df.shape[1]} colonnes")

limit_val = None if int(limit) == 0 else int(limit)

try:
    prod_X, prod_meta = read_prod_from_db(db_url=db_url, limit=limit_val, endpoint=endpoint)
except Exception as e:
    st.error(f"Erreur DB: {e}")
    st.stop()

if len(prod_meta) == 0:
    st.warning("Aucune ligne trouvée en base (prod_requests).")
    st.stop()

st.success(f"DB chargée: {len(prod_meta)} requêtes | prod inputs: {prod_X.shape[0]}×{prod_X.shape[1]}")

# --- OPS metrics
st.subheader("1) Santé API (ops)")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Nb requêtes", int(len(prod_meta)))
with c2:
    st.metric("Taux succès (200)", float((prod_meta["status_code"] == 200).mean() * 100), "%")
with c3:
    st.metric("Latence médiane (ms)", float(pd.to_numeric(prod_meta["latency_ms"], errors="coerce").median()))

lat = pd.to_numeric(prod_meta["latency_ms"], errors="coerce")
st.plotly_chart(px.histogram(lat.dropna(), nbins=30, title="Distribution latence (ms)"), use_container_width=True)

status_counts = prod_meta["status_code"].value_counts(dropna=False).sort_index()
st.plotly_chart(px.bar(status_counts, title="Codes HTTP"), use_container_width=True)

# --- Drift
st.subheader("2) Data Drift (PSI)")

common_cols = [c for c in ref_df.columns if c in prod_X.columns]
missing_in_prod = [c for c in ref_df.columns if c not in prod_X.columns]

if missing_in_prod:
    st.warning(f"{len(missing_in_prod)} features de référence absentes en prod (ex: {missing_in_prod[:5]})")

rows = []
for col in common_cols:
    ref_s = ref_df[col]
    prod_s = prod_X[col]

    if ref_s.dtype == "object":
        v = psi_categorical(ref_s, prod_s)
        kind = "categorical"
    else:
        v = psi(ref_s, prod_s, bins=bins)
        kind = "numeric"

    rows.append({"feature": col, "psi": v, "type": kind})

drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
st.dataframe(drift_df, use_container_width=True, hide_index=True)

topk = drift_df.dropna().head(20)
st.plotly_chart(px.bar(topk[::-1], x="psi", y="feature", orientation="h", title="Top 20 PSI (drift)"),
                use_container_width=True)

# --- Inspect feature
st.subheader("3) Détail feature")

feat = st.selectbox("Choisir une feature", common_cols)
ref_s = ref_df[feat]
prod_s = prod_X[feat]

colA, colB = st.columns(2)
with colA:
    st.write("Référence")
    st.write(ref_s.describe(include="all"))
with colB:
    st.write("Production")
    st.write(prod_s.describe(include="all"))

if ref_s.dtype == "object":
    ref_counts = ref_s.fillna("__MISSING__").astype(str).value_counts().head(20)
    prod_counts = prod_s.fillna("__MISSING__").astype(str).value_counts().head(20)
    merged = pd.DataFrame({"ref": ref_counts, "prod": prod_counts}).fillna(0).reset_index().rename(columns={"index": feat})
    fig = px.bar(merged, x=feat, y=["ref", "prod"], barmode="group", title=f"Top catégories: {feat}")
else:
    fig = px.histogram(
        pd.DataFrame({
            "ref": pd.to_numeric(ref_s, errors="coerce"),
            "prod": pd.to_numeric(prod_s, errors="coerce")
        }),
        x=["ref", "prod"],
        nbins=30,
        barmode="overlay",
        title=f"Distribution (overlay): {feat}",
        opacity=0.6,
    )
st.plotly_chart(fig, use_container_width=True)