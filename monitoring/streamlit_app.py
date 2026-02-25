from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from core.db.conn import init_db
from core.db.repo_prod_requests import select_prod_requests
from core.db.repo_ref_dist import load_all_ref, load_one_ref

load_dotenv()
init_db()

# --- Sécurité / RGPD : colonnes à ne jamais exposer dans le dashboard
EXCLUDED_FEATURES = {"SK_ID_CURR"}      # drift + détails
EXCLUDED_META_COLS = {"sk_id_curr"}     # meta affichée (logs)

# -----------------------
# PSI helpers (ref dist en DB)
# -----------------------

def psi_from_dists(ref_p: np.ndarray, prod_p: np.ndarray, eps: float = 1e-6) -> float:
    r = np.clip(ref_p, eps, 1)
    p = np.clip(prod_p, eps, 1)
    return float(np.sum((p - r) * np.log(p / r)))


def prod_dist_numeric(prod_s: pd.Series, edges: List[float], labels: List[str]) -> Tuple[List[str], np.ndarray]:
    x = pd.to_numeric(prod_s, errors="coerce").dropna()
    if len(x) == 0:
        return labels, np.zeros(len(labels), dtype=float)

    # pd.cut avec edges
    b = pd.cut(x, bins=np.array(edges, dtype=float), include_lowest=True, duplicates="drop")
    dist = b.value_counts(normalize=True).sort_index()
    prod_map = {str(idx): float(v) for idx, v in dist.items()}

    p = np.array([prod_map.get(lab, 0.0) for lab in labels], dtype=float)
    return labels, p


def prod_dist_categorical(prod_s: pd.Series, labels_ref: List[str]) -> Tuple[List[str], np.ndarray]:
    x = prod_s.fillna("__MISSING__").astype(str)

    # si ref contient __OTHER__, on y met tout ce qui n’est pas dans labels_ref
    has_other = "__OTHER__" in labels_ref
    allowed = set(labels_ref)

    mapped = []
    for v in x.tolist():
        if v in allowed:
            mapped.append(v)
        else:
            mapped.append("__OTHER__" if has_other else v)

    vc = pd.Series(mapped).value_counts(normalize=True)
    prod_map = {k: float(v) for k, v in vc.items()}

    p = np.array([prod_map.get(lab, 0.0) for lab in labels_ref], dtype=float)
    return labels_ref, p


# -----------------------
# UI
# -----------------------

st.set_page_config(page_title="PAD — Monitoring", layout="wide")
st.title("Monitoring — API Ops + Data Drift (référence en DB)")

default_db_url = os.getenv("DATABASE_URL", "")
with st.sidebar:
    st.header("Inputs")
    endpoint = st.text_input("Endpoint", "/predict")
    limit = st.number_input("Nb requêtes (0=all)", min_value=0, value=1000, step=100)
    st.caption("PSI: <0.1 OK | 0.1–0.25 à surveiller | >0.25 drift fort")

limit_val = None if int(limit) == 0 else int(limit)

rows = select_prod_requests(endpoint=endpoint, limit=limit_val or 1000000)
if not rows:
    st.warning("Aucune requête trouvée en DB (prod_requests).")
    st.stop()

prod_meta = pd.DataFrame([
    {k: r.get(k) for k in ["ts","endpoint","status_code","latency_ms","error","message"]}
    for r in rows
])
prod_inputs = pd.DataFrame([r.get("inputs") or {} for r in rows])
# Sécurité : on retire les features sensibles du dataset prod utilisé par le dashboard
prod_inputs = prod_inputs.drop(columns=[c for c in EXCLUDED_FEATURES if c in prod_inputs.columns], errors="ignore")
st.success(f"DB chargée: {len(prod_meta)} requêtes | inputs: {prod_inputs.shape[0]}×{prod_inputs.shape[1]}")

# ---- OPS
st.subheader("1) Santé API (ops)")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Nb requêtes", int(len(prod_meta)))
with c2:
    st.metric("Taux succès (200)", round(float((prod_meta["status_code"] == 200).mean() * 100), 2), "%")
with c3:
    lat = pd.to_numeric(prod_meta["latency_ms"], errors="coerce")
    st.metric("Latence médiane (ms)", float(lat.median()) if lat.notna().any() else 0.0)

st.plotly_chart(px.histogram(lat.dropna(), nbins=30, title="Distribution latence (ms)"), use_container_width=True)
status_counts = prod_meta["status_code"].value_counts(dropna=False).sort_index()
st.plotly_chart(px.bar(status_counts, title="Codes HTTP"), use_container_width=True)

# ---- DRIFT (ref en DB)
st.subheader("2) Data Drift (PSI) — référence en DB")

ref_rows = load_all_ref()
if not ref_rows:
    st.warning("Aucune référence en DB (ref_feature_dist). Lance build_reference_dist.py d’abord.")
    st.stop()

ref_df = pd.DataFrame(ref_rows)
ref_features = [f for f in ref_df["feature"].tolist() if f not in EXCLUDED_FEATURES]

common = [c for c in ref_features if c in prod_inputs.columns]
missing_in_prod = [c for c in ref_features if c not in prod_inputs.columns]
if missing_in_prod:
    st.warning(f"{len(missing_in_prod)} features de référence absentes en prod (ex: {missing_in_prod[:5]})")

psi_rows: List[Dict[str, Any]] = []
ref_map = {r["feature"]: r for r in ref_rows}

for feat in common:
    if feat in EXCLUDED_FEATURES:
        continue
    ref = ref_map[feat]
    kind = ref["kind"]
    ref_dist = ref["ref_dist_json"] or {}
    labels_ref = (ref_dist.get("labels") or [])
    ref_p = np.array(ref_dist.get("p") or [], dtype=float)

    prod_s = prod_inputs[feat] if feat in prod_inputs.columns else pd.Series([], dtype=float)

    if kind == "numeric":
        bins = ref.get("bins_json") or {}
        edges = bins.get("edges") or []
        if not edges or not labels_ref or ref_p.size == 0:
            v = np.nan
        else:
            _, prod_p = prod_dist_numeric(prod_s, edges=edges, labels=labels_ref)
            v = psi_from_dists(ref_p, prod_p)
    else:
        if not labels_ref or ref_p.size == 0:
            v = np.nan
        else:
            _, prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref)
            v = psi_from_dists(ref_p, prod_p)

    psi_rows.append({"feature": feat, "psi": v, "type": kind})

drift = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
st.dataframe(drift, use_container_width=True, hide_index=True)

topk = drift.dropna().head(20)
st.plotly_chart(px.bar(topk[::-1], x="psi", y="feature", orientation="h", title="Top 20 PSI (drift)"),
                use_container_width=True)

# ---- Détail feature
st.subheader("3) Détail feature (ref vs prod)")
feat = st.selectbox("Choisir une feature", common)

ref_one = load_one_ref(feat)
if ref_one is None:
    st.warning("Référence introuvable pour cette feature.")
    st.stop()

ref_dist = ref_one["ref_dist_json"] or {}
labels_ref = ref_dist.get("labels") or []
ref_p = np.array(ref_dist.get("p") or [], dtype=float)

prod_s = prod_inputs[feat]
kind = ref_one["kind"]

colA, colB = st.columns(2)
with colA:
    st.write("Référence (dist en DB)")
    st.write(pd.DataFrame({"label": labels_ref, "p_ref": ref_p}))
with colB:
    st.write("Production (stats brutes)")
    st.write(prod_s.describe(include="all"))

if kind == "numeric":
    edges = (ref_one.get("bins_json") or {}).get("edges") or []
    _, prod_p = prod_dist_numeric(prod_s, edges=edges, labels=labels_ref)
    df_plot = pd.DataFrame({"label": labels_ref, "ref": ref_p, "prod": prod_p})
    fig = px.bar(df_plot, x="label", y=["ref", "prod"], barmode="group", title=f"Dist: {feat}")
else:
    _, prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref)
    df_plot = pd.DataFrame({"label": labels_ref, "ref": ref_p, "prod": prod_p})
    fig = px.bar(df_plot, x="label", y=["ref", "prod"], barmode="group", title=f"Dist: {feat}")

st.plotly_chart(fig, use_container_width=True)