from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------
# Utils
# -----------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_prod_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    On extrait les inputs 'nettoyés' (inputs) si présents, sinon (request).
    Selon ton log_event tu as: inputs et outputs (en success).
    """
    feats = []
    meta = []
    for e in events:
        inp = e.get("inputs") or e.get("request") or {}
        feats.append(inp)
        meta.append({
            "ts": e.get("ts"),
            "endpoint": e.get("endpoint"),
            "status_code": e.get("status_code"),
            "latency_ms": e.get("latency_ms"),
            "error": e.get("error"),
            "message": e.get("message"),
        })

    X = pd.DataFrame(feats)
    M = pd.DataFrame(meta)
    return X, M

def psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """
    PSI pour variables numériques.
    - On calcule les bins sur ref (quantiles), puis on compare les proportions.
    """
    ref = ref.dropna()
    prod = prod.dropna()
    if len(ref) < 50 or len(prod) < 50:
        return np.nan

    # Si variable quasi constante
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

        # align
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

    # union catégories
    idx = ref_dist.index.union(prod_dist.index)
    ref_dist = ref_dist.reindex(idx).fillna(0)
    prod_dist = prod_dist.reindex(idx).fillna(0)

    eps = 1e-6
    r = np.clip(ref_dist.values, eps, 1)
    p = np.clip(prod_dist.values, eps, 1)
    return float(np.sum((p - r) * np.log(p / r)))

def is_categorical_feature(name: str, cat_features: set[str]) -> bool:
    return name in cat_features

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Prêt à Dépenser — Monitoring", layout="wide")

st.title("Monitoring — API & Data Drift (PoC local)")

base_dir = Path(__file__).resolve().parent
default_ref = Path.cwd() / "data" / "processed" / "train_split_final.csv" 
default_logs = Path.cwd() / "prod_logs" / "requests.jsonl"  # utile si tu lances depuis la racine

with st.sidebar:
    st.header("Inputs")
    ref_path = st.text_input("Reference CSV", str(default_ref))
    logs_path = st.text_input("Prod logs JSONL", str(default_logs))
    bins = st.slider("PSI bins (numériques)", 5, 20, 10)
    st.caption("Seuils PSI usuels: <0.1 OK | 0.1–0.25 à surveiller | >0.25 drift fort")

ref_path = Path(ref_path)
logs_path = Path(logs_path)

if not ref_path.exists():
    st.error(f"Reference introuvable: {ref_path}")
    st.stop()

ref_df = pd.read_csv(ref_path)
st.success(f"Reference chargée: {ref_df.shape[0]} lignes × {ref_df.shape[1]} colonnes")

events = read_jsonl(logs_path)
if len(events) == 0:
    st.warning(f"Aucun log trouvé (ou fichier absent): {logs_path}")
    st.stop()

prod_X, prod_meta = build_prod_dataframe(events)
st.success(f"Logs chargés: {len(events)} événements | prod inputs: {prod_X.shape[0]}×{prod_X.shape[1]}")

# --- OPS metrics (latence & erreurs)
st.subheader("1) Santé API (ops)")
c1, c2, c3 = st.columns(3)
status_counts = prod_meta["status_code"].value_counts(dropna=False)

with c1:
    st.metric("Nb requêtes", int(len(prod_meta)))
with c2:
    st.metric("Taux succès (200)", float((prod_meta["status_code"] == 200).mean() * 100), "%")
with c3:
    st.metric("Latence médiane (ms)", float(pd.to_numeric(prod_meta["latency_ms"], errors="coerce").median()))

lat = pd.to_numeric(prod_meta["latency_ms"], errors="coerce")
fig_lat = px.histogram(lat.dropna(), nbins=30, title="Distribution latence (ms)")
st.plotly_chart(fig_lat, use_container_width=True)

fig_status = px.bar(status_counts.sort_index(), title="Codes HTTP")
st.plotly_chart(fig_status, use_container_width=True)

# --- Drift
st.subheader("2) Data Drift (PSI)")
common_cols = [c for c in ref_df.columns if c in prod_X.columns]
missing_in_prod = [c for c in ref_df.columns if c not in prod_X.columns]

if missing_in_prod:
    st.warning(f"{len(missing_in_prod)} features de référence absentes en prod (ex: {missing_in_prod[:5]})")

# Optionnel: si tu as la liste des cat features quelque part, tu peux la charger.
# Ici: on considère "cat_features" vide par défaut.
cat_features = set()

rows = []
for col in common_cols:
    ref_s = ref_df[col]
    prod_s = prod_X[col]
    if is_categorical_feature(col, cat_features) or ref_s.dtype == "object":
        v = psi_categorical(ref_s, prod_s)
        kind = "categorical"
    else:
        v = psi(ref_s, prod_s, bins=bins)
        kind = "numeric"
    rows.append({"feature": col, "psi": v, "type": kind})

drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
st.dataframe(drift_df, use_container_width=True, hide_index=True)

# Top features chart
topk = drift_df.dropna().head(20)
fig_top = px.bar(topk[::-1], x="psi", y="feature", orientation="h", title="Top 20 PSI (drift)")
st.plotly_chart(fig_top, use_container_width=True)

# Inspect 1 feature
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

# Plot
if ref_s.dtype == "object":
    ref_counts = ref_s.fillna("__MISSING__").astype(str).value_counts().head(20)
    prod_counts = prod_s.fillna("__MISSING__").astype(str).value_counts().head(20)
    merged = pd.DataFrame({"ref": ref_counts, "prod": prod_counts}).fillna(0).reset_index().rename(columns={"index": feat})
    fig = px.bar(merged, x=feat, y=["ref", "prod"], barmode="group", title=f"Top catégories: {feat}")
else:
    fig = px.histogram(
        pd.DataFrame({"ref": pd.to_numeric(ref_s, errors="coerce"), "prod": pd.to_numeric(prod_s, errors="coerce")}),
        x=["ref", "prod"],
        nbins=30,
        barmode="overlay",
        title=f"Distribution (overlay): {feat}",
        opacity=0.6,
    )
st.plotly_chart(fig, use_container_width=True)