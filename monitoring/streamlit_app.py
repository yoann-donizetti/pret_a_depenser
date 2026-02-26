from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# --- Path / project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))  # pour import core.*
sys.path.append(str(PROJECT_ROOT / "monitoring"))  # pour import monitoring.lib.*

from core.db.conn import init_db
from core.db.repo_prod_requests import select_prod_requests
from core.db.repo_ref_dist import load_all_ref, load_one_ref

from monitoring.lib.filters import apply_time_filter
from monitoring.lib.ops import latency_stats_ms, error_rate
from monitoring.lib.drift import (
    psi_from_dists,
    prod_dist_numeric,
    prod_dist_categorical,
    count_drift,
)
from monitoring.lib.timings import extract_timings

# -----------------------
# Setup
# -----------------------
load_dotenv()
init_db()

# --- Sécurité / RGPD : colonnes à ne jamais exposer dans le dashboard
EXCLUDED_FEATURES = {"SK_ID_CURR"}  # drift + détails
EXCLUDED_META_COLS = {"sk_id_curr"}  # meta affichée (logs)

st.set_page_config(page_title="PAD — Monitoring", layout="wide")
st.title("Monitoring — API Ops + Data Drift (référence en DB)")

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Inputs")
    endpoint = st.text_input("Endpoint", "/predict")
    limit = st.number_input("Nb requêtes (0=all)", min_value=0, value=1000, step=100)

    time_window = st.selectbox("Fenêtre temporelle", ["all", "24h", "7d", "30d"], index=0)
    p95_threshold = st.number_input("Seuil p95 (ms) warning", min_value=1, value=200, step=10)

    st.caption("PSI: <0.1 OK | 0.1–0.25 à surveiller | >0.25 drift fort")

limit_val = None if int(limit) == 0 else int(limit)

# -----------------------
# Load logs from DB
# -----------------------
rows = select_prod_requests(endpoint=endpoint, limit=limit_val or 1_000_000)
if not rows:
    st.warning("Aucune requête trouvée en DB (prod_requests).")
    st.stop()

# Meta df (sans sk_id_curr)
prod_meta = pd.DataFrame(
    [{k: r.get(k) for k in ["ts", "endpoint", "status_code", "latency_ms", "error", "message"]} for r in rows]
)

# Filtre temporel sur meta, puis on refiltre rows pour garder inputs/outputs cohérents
prod_meta = apply_time_filter(prod_meta, time_window)

kept_ts = set(prod_meta["ts"].astype(str).tolist())
rows = [r for r in rows if str(r.get("ts")) in kept_ts]

prod_inputs = pd.DataFrame([r.get("inputs") or {} for r in rows])
prod_outputs = pd.DataFrame([r.get("outputs") or {} for r in rows])
timing_df = extract_timings(prod_outputs)
# Sécurité : on retire les features sensibles du dataset prod utilisé par le dashboard
prod_inputs = prod_inputs.drop(columns=[c for c in EXCLUDED_FEATURES if c in prod_inputs.columns], errors="ignore")

st.success(f"DB chargée: {len(prod_meta)} requêtes | inputs: {prod_inputs.shape[0]}×{prod_inputs.shape[1]}")

# -----------------------
# 1) OPS
# -----------------------
st.subheader("1) Santé API (ops)")

lat = pd.to_numeric(prod_meta["latency_ms"], errors="coerce")
stats = latency_stats_ms(lat)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Nb requêtes", int(len(prod_meta)))
with c2:
    st.metric("Taux succès (200)", round(float((prod_meta["status_code"] == 200).mean() * 100), 2), "%")
with c3:
    st.metric("Taux erreur (>=400)", round(error_rate(prod_meta["status_code"]), 2), "%")
with c4:
    st.metric("Latence p95 (ms)", round(stats["p95"], 2))

if stats["p95"] > float(p95_threshold):
    st.warning(f" p95 ({stats['p95']:.2f} ms) dépasse le seuil ({p95_threshold} ms).")
else:
    st.success(f" p95 ({stats['p95']:.2f} ms) sous le seuil ({p95_threshold} ms).")

#timings détaillés

st.subheader("Timings détaillés (ms) — DB / Validation / Inference / Total")

if timing_df.empty:
    st.info("Aucun champ 'timing' trouvé dans outputs (logs).")
else:
    c1, c2, c3, c4 = st.columns(4)

    def q95(s):
        s = s.dropna()
        return float(s.quantile(0.95)) if len(s) else 0.0

    with c1:
        st.metric("DB p95", round(q95(timing_df["db_ms"]), 2))
    with c2:
        st.metric("Validation p95", round(q95(timing_df["validation_ms"]), 2))
    with c3:
        st.metric("Inference p95", round(q95(timing_df["inference_ms"]), 2))
    with c4:
        st.metric("Total p95", round(q95(timing_df["total_ms"]), 2))

    st.plotly_chart(
        px.histogram(timing_df["db_ms"].dropna(), nbins=30, title="DB time (ms)"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.histogram(timing_df["inference_ms"].dropna(), nbins=30, title="Inference time (ms)"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.histogram(timing_df["validation_ms"].dropna(), nbins=30, title="Validation time (ms)"),
        use_container_width=True,
    )




st.plotly_chart(px.histogram(lat.dropna(), nbins=30, title="Distribution latence totale (ms)"), use_container_width=True)

status_counts = prod_meta["status_code"].value_counts(dropna=False).sort_index()
st.plotly_chart(px.bar(status_counts, title="Codes HTTP"), use_container_width=True)

# Distribution des décisions (si logguée)
st.subheader("Décisions (prod)")
if "decision" in prod_outputs.columns:
    dec = prod_outputs["decision"].fillna("UNKNOWN").astype(str)
    dec_counts = dec.value_counts()
    st.plotly_chart(
        px.pie(dec_counts, values=dec_counts.values, names=dec_counts.index, title="ACCEPTED / REFUSED"),
        use_container_width=True,
    )
else:
    st.info("Aucune colonne 'decision' trouvée dans outputs (logs).")

# -----------------------
# 2) Drift (PSI)
# -----------------------
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

ref_map = {r["feature"]: r for r in ref_rows}

psi_rows: List[Dict[str, Any]] = []
for feat in common:
    if feat in EXCLUDED_FEATURES:
        continue

    ref = ref_map[feat]
    kind = ref["kind"]
    ref_dist = ref["ref_dist_json"] or {}
    labels_ref = (ref_dist.get("labels") or [])
    ref_p = np.array(ref_dist.get("p") or [], dtype=float)

    prod_s = prod_inputs[feat]

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

# Compteur PSI > 0.25
n_drift = count_drift(drift, threshold=0.25)
st.metric("Nb features PSI > 0.25", n_drift)

st.dataframe(drift, use_container_width=True, hide_index=True)

# Top 20 PSI (barre horizontale, plus grand en haut)
topk = drift.dropna().head(20).copy()
topk = topk.sort_values("psi", ascending=False)

fig_top = px.bar(topk, x="psi", y="feature", orientation="h", title="Top 20 PSI (drift)")
fig_top.update_yaxes(autorange="reversed")
fig_top.update_layout(height=650, margin=dict(l=260, r=40, t=60, b=40))
st.plotly_chart(fig_top, use_container_width=True)

# -----------------------
# 3) Détail feature (ref vs prod)
# -----------------------
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
