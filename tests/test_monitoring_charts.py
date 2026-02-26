import pandas as pd
import plotly.graph_objects as go

from monitoring.lib.charts import (
    hist_latency,
    bar_status_codes,
    pie_decisions,
    bar_top_drift,
    bar_ref_vs_prod,
)


def test_hist_latency_returns_figure():
    s = pd.Series([10, 20, 30])
    fig = hist_latency(s, "Latency")
    assert isinstance(fig, go.Figure)


def test_bar_status_codes_returns_figure():
    s = pd.Series([3, 1], index=[200, 500])
    fig = bar_status_codes(s, "HTTP")
    assert isinstance(fig, go.Figure)


def test_pie_decisions_returns_figure():
    s = pd.Series([10, 5], index=["ACCEPTED", "REFUSED"])
    fig = pie_decisions(s, "Decisions")
    assert isinstance(fig, go.Figure)


def test_bar_top_drift_returns_figure():
    df = pd.DataFrame({"feature": ["a", "b"], "psi": [0.3, 0.1]})
    fig = bar_top_drift(df, "Top drift")
    assert isinstance(fig, go.Figure)


def test_bar_ref_vs_prod_returns_figure():
    df = pd.DataFrame({"label": ["x", "y"], "ref": [0.6, 0.4], "prod": [0.5, 0.5]})
    fig = bar_ref_vs_prod(df, "feat")
    assert isinstance(fig, go.Figure)