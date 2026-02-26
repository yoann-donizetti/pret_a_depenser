from __future__ import annotations

import pandas as pd
import plotly.express as px


def hist_latency(lat: pd.Series, title: str, nbins: int = 30):
    return px.histogram(pd.to_numeric(lat, errors="coerce").dropna(), nbins=nbins, title=title)


def bar_status_codes(status_counts: pd.Series, title: str = "Codes HTTP"):
    return px.bar(status_counts, title=title)


def pie_decisions(dec_counts: pd.Series, title: str = "Décisions"):
    return px.pie(dec_counts, values=dec_counts.values, names=dec_counts.index, title=title)


def bar_top_drift(topk_df: pd.DataFrame, title: str = "Top PSI (drift)"):
    fig = px.bar(topk_df, x="psi", y="feature", orientation="h", title=title)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=650, margin=dict(l=260, r=40, t=60, b=40))
    return fig


def bar_ref_vs_prod(df_plot: pd.DataFrame, feature_name: str):
    return px.bar(df_plot, x="label", y=["ref", "prod"], barmode="group", title=f"Dist: {feature_name}")