from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- Path / project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))  # import core.*
sys.path.append(str(PROJECT_ROOT / "monitoring"))  # import monitoring.*

from core.db.conn import init_db

from monitoring.lib.constants import DEFAULTS, TIME_WINDOWS, PSI_THRESHOLDS
from monitoring.lib.security import EXCLUDED_FEATURES
from monitoring.lib.data import load_prod_data, load_reference, load_reference_one
from monitoring.lib.ops import latency_stats_ms, error_rate, success_rate
from monitoring.lib.timings import extract_timings, compute_timing_stats
from monitoring.lib.drift import (
    compute_drift_table,
    count_drift,
    prod_dist_numeric,
    prod_dist_categorical,
    psi_from_dists,
)
from monitoring.lib.charts import (
    hist_latency,
    bar_status_codes,
    pie_decisions,
    bar_top_drift,
    bar_ref_vs_prod,
)

# -----------------------
# Setup
# -----------------------
load_dotenv()
init_db()

st.set_page_config(page_title="PAD — Monitoring", layout="wide")
st.title("Monitoring — API Ops + Data Drift (référence en DB)")

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Inputs")
    endpoint = st.text_input("Endpoint", DEFAULTS["endpoint"])
    limit = st.number_input("Nb requêtes (0=all)", min_value=0, value=int(DEFAULTS["limit"]), step=100)

    time_window = st.selectbox("Fenêtre temporelle", TIME_WINDOWS, index=0)
    p95_threshold = st.number_input(
        "Seuil p95 total (ms) warning",
        min_value=1,
        value=int(DEFAULTS["p95_threshold_ms"]),
        step=10,
    )

    drift_threshold = st.number_input(
        "Seuil drift PSI",
        min_value=0.01,
        value=float(DEFAULTS["drift_threshold"]),
        step=0.01,
    )

    st.caption(
        f"PSI: <{PSI_THRESHOLDS['ok']} OK | {PSI_THRESHOLDS['ok']}–{PSI_THRESHOLDS['watch']} à surveiller | >{PSI_THRESHOLDS['watch']} drift fort"
    )

limit_val = None if int(limit) == 0 else int(limit)

# -----------------------
# Load prod data
# -----------------------
prod_meta, prod_inputs, prod_outputs, rows = load_prod_data(
    endpoint=endpoint,
    limit=limit_val,
    time_window=time_window,
    excluded_features=EXCLUDED_FEATURES,
)

if prod_meta.empty:
    st.warning("Aucune requête trouvée en DB (prod_requests) ou DB non accessible.")
    st.stop()

st.success(f"DB chargée: {len(prod_meta)} requêtes | inputs: {prod_inputs.shape[0]}×{prod_inputs.shape[1]}")

# -----------------------
# OPS
# -----------------------
st.subheader("1) Santé API (ops)")

lat_total = pd.to_numeric(prod_meta["latency_ms"], errors="coerce")
stats_total = latency_stats_ms(lat_total)

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Nb requêtes", int(len(prod_meta)))
with c2:
    st.metric("Succès (200)", round(success_rate(prod_meta["status_code"]), 2), "%")
with c3:
    st.metric("Erreurs (>=400)", round(error_rate(prod_meta["status_code"]), 2), "%")
with c4:
    st.metric("Total p95 (ms)", round(stats_total["p95"], 2))

# p50 / p95 / p99 (ligne)
st.markdown("**Latence totale (ms)**")
c50, c95, c99, c_mean, c_median = st.columns(5)
c50.metric("p50", round(stats_total["p50"], 2))
c95.metric("p95", round(stats_total["p95"], 2))
c99.metric("p99", round(stats_total["p99"], 2))
c_mean.metric("mean", round(stats_total["mean"], 2))
c_median.metric("median", round(stats_total["median"], 2))  

if float(stats_total["p95"]) > float(p95_threshold):
    st.warning(f"Total p95 ({stats_total['p95']:.2f} ms) dépasse le seuil ({p95_threshold} ms).")
else:
    st.success(f"Total p95 ({stats_total['p95']:.2f} ms) sous le seuil ({p95_threshold} ms).")

st.plotly_chart(hist_latency(lat_total, "Distribution latence totale (ms)"), use_container_width=True)

status_counts = prod_meta["status_code"].value_counts(dropna=False).sort_index()
st.plotly_chart(bar_status_codes(status_counts, "Codes HTTP"), use_container_width=True)

# -----------------------
# Timings détaillés
# -----------------------
st.subheader("Timings détaillés (ms) — DB / Validation / Inference / Total")

timing_df = extract_timings(prod_outputs)
timing_stats = compute_timing_stats(timing_df)

if not timing_stats:
    st.info("Aucun champ 'timing' trouvé dans outputs (logs).")
else:
    # p50/p95/p99 en ligne par composant
    st.markdown("**DB (ms)**")
    c1, c2, c3,c4,c5 = st.columns(5)
    c1.metric("p50", round(timing_stats["db_ms"]["p50"], 2))
    c2.metric("p95", round(timing_stats["db_ms"]["p95"], 2))
    c3.metric("p99", round(timing_stats["db_ms"]["p99"], 2))
    c4.metric("mean", round(timing_stats["db_ms"]["mean"], 2))
    c5.metric("median", round(timing_stats["db_ms"]["median"], 2))

    st.markdown("**Validation (ms)**")
    c1, c2, c3,c4,c5 = st.columns(5)
    c1.metric("p50", round(timing_stats["validation_ms"]["p50"], 2))
    c2.metric("p95", round(timing_stats["validation_ms"]["p95"], 2))
    c3.metric("p99", round(timing_stats["validation_ms"]["p99"], 2))
    c4.metric("mean", round(timing_stats["validation_ms"]["mean"], 2))
    c5.metric("median", round(timing_stats["validation_ms"]["median"], 2))

    st.markdown("**Inference (ms)**")
    c1, c2, c3,c4,c5 = st.columns(5)
    c1.metric("p50", round(timing_stats["inference_ms"]["p50"], 2))
    c2.metric("p95", round(timing_stats["inference_ms"]["p95"], 2))
    c3.metric("p99", round(timing_stats["inference_ms"]["p99"], 2))
    c4.metric("mean", round(timing_stats["inference_ms"]["mean"], 2))
    c5.metric("median", round(timing_stats["inference_ms"]["median"], 2))

    st.markdown("**Total (timing) (ms)**")
    c1, c2, c3,c4,c5 = st.columns(5)
    c1.metric("p50", round(timing_stats["total_ms"]["p50"], 2))
    c2.metric("p95", round(timing_stats["total_ms"]["p95"], 2))
    c3.metric("p99", round(timing_stats["total_ms"]["p99"], 2))
    c4.metric("mean", round(timing_stats["total_ms"]["mean"], 2))
    c5.metric("median", round(timing_stats["total_ms"]["median"], 2))
    # charts
    st.plotly_chart(hist_latency(timing_df["db_ms"], "DB time (ms)"), use_container_width=True)
    st.plotly_chart(hist_latency(timing_df["inference_ms"], "Inference time (ms)"), use_container_width=True)
    st.plotly_chart(hist_latency(timing_df["validation_ms"], "Validation time (ms)"), use_container_width=True)
    st.plotly_chart(hist_latency(timing_df["total_ms"], "Total (timing) (ms)"), use_container_width=True)

# -----------------------
# Décisions
# -----------------------
st.subheader("Décisions (prod)")

if "decision" in prod_outputs.columns:
    dec = prod_outputs["decision"].fillna("UNKNOWN").astype(str)
    dec_counts = dec.value_counts()
    st.plotly_chart(pie_decisions(dec_counts, "ACCEPTED / REFUSED"), use_container_width=True)
else:
    st.info("Aucune colonne 'decision' trouvée dans outputs (logs).")

# -----------------------
# Drift PSI
# -----------------------
st.subheader("2) Data Drift (PSI) — référence en DB")

ref_rows = load_reference()
if not ref_rows:
    st.warning("Aucune référence en DB (ref_feature_dist). Lance build_reference_dist.py d’abord.")
    st.stop()

drift_df = compute_drift_table(prod_inputs=prod_inputs, ref_rows=ref_rows, excluded_features=EXCLUDED_FEATURES)

n_drift = count_drift(drift_df, threshold=float(drift_threshold))
st.metric(f"Nb features PSI > {drift_threshold}", n_drift)

st.dataframe(drift_df, use_container_width=True, hide_index=True)

topk = drift_df.dropna().head(int(DEFAULTS["topk_drift"])).copy()
topk = topk.sort_values("psi", ascending=False)
st.plotly_chart(bar_top_drift(topk, f"Top {DEFAULTS['topk_drift']} PSI (drift)"), use_container_width=True)

# -----------------------
# Détail feature
# -----------------------
st.subheader("3) Détail feature (ref vs prod)")

ref_features = [r["feature"] for r in ref_rows if r.get("feature") not in EXCLUDED_FEATURES]
common = [c for c in ref_features if c in prod_inputs.columns]

feat = st.selectbox("Choisir une feature", common)

ref_one = load_reference_one(feat)
if ref_one is None:
    st.warning("Référence introuvable pour cette feature.")
    st.stop()

ref_dist = ref_one.get("ref_dist_json") or {}
labels_ref = ref_dist.get("labels") or []
ref_p = np.array(ref_dist.get("p") or [], dtype=float)
kind = ref_one.get("kind")

prod_s = prod_inputs[feat]

colA, colB = st.columns(2)
with colA:
    st.write("Référence (dist en DB)")
    st.write(pd.DataFrame({"label": labels_ref, "p_ref": ref_p}))
with colB:
    st.write("Production (stats brutes)")
    st.write(prod_s.describe(include="all"))

prod_p = np.zeros(len(labels_ref), dtype=float)

if kind == "numeric":
    edges = (ref_one.get("bins_json") or {}).get("edges") or []
    if edges and labels_ref and ref_p.size:
        _, prod_p = prod_dist_numeric(prod_s, edges=edges, labels=labels_ref)
else:
    if labels_ref and ref_p.size:
        _, prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref)

df_plot = pd.DataFrame({"label": labels_ref, "ref": ref_p, "prod": prod_p})
st.plotly_chart(bar_ref_vs_prod(df_plot, feat), use_container_width=True)

psi_val = float("nan")
if ref_p.size and prod_p.size and len(ref_p) == len(prod_p):
    psi_val = psi_from_dists(ref_p, prod_p)

st.caption(f"PSI({feat}) = {psi_val:.4f}" if psi_val == psi_val else "PSI non calculable (référence/production incomplète).")