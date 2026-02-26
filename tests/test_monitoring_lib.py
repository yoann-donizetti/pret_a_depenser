import numpy as np
import pandas as pd

from monitoring.lib.ops import latency_stats_ms, success_rate, error_rate
from monitoring.lib.timings import extract_timings, compute_timing_stats
from monitoring.lib.drift import (
    psi_from_dists,
    prod_dist_numeric,
    prod_dist_categorical,
    count_drift,
    compute_drift_table,
)
from monitoring.lib.constants import DEFAULTS, TIME_WINDOWS, PSI_THRESHOLDS
from monitoring.lib.security import EXCLUDED_FEATURES
from monitoring.lib.filters import apply_time_filter


# -----------------------
# OPS
# -----------------------

def test_latency_stats_ms_empty():
    s = pd.Series([], dtype=float)
    out = latency_stats_ms(s)
    assert out["p50"] == 0.0
    assert out["p95"] == 0.0
    assert out["p99"] == 0.0
    assert out["mean"] == 0.0



def test_latency_stats_ms_non_empty():
    s = pd.Series([10, 20, 30, 40, 50], dtype=float)
    out = latency_stats_ms(s)
    assert out["p50"] > 0
    assert out["p95"] >= out["p50"]
    assert out["p99"] >= out["p95"]
    assert out["mean"] > 0


def test_success_rate_and_error_rate():
    status = pd.Series([200, 200, 404, 500])
    assert success_rate(status) == 50.0
    assert error_rate(status) == 50.0


# -----------------------
# TIMINGS
# -----------------------

def test_extract_timings_ok():
    outputs = pd.DataFrame(
        {
            "timing": [
                {"db_ms": 10, "validation_ms": 1, "inference_ms": 20, "total_ms": 31},
                {"db_ms": 12, "validation_ms": 2, "inference_ms": 30, "total_ms": 44},
            ]
        }
    )
    tdf = extract_timings(outputs)
    assert list(tdf.columns) == ["db_ms", "validation_ms", "inference_ms", "total_ms"]
    assert len(tdf) == 2


def test_extract_timings_missing_column():
    outputs = pd.DataFrame({"decision": ["A", "B"]})
    tdf = extract_timings(outputs)
    assert tdf.empty


def test_compute_timing_stats_ok():
    tdf = pd.DataFrame(
        {
            "db_ms": [10, 20, 30],
            "validation_ms": [1, 2, 3],
            "inference_ms": [50, 60, 70],
            "total_ms": [61, 82, 103],
        }
    )
    stats = compute_timing_stats(tdf)
    assert "db_ms" in stats and "p95" in stats["db_ms"]
    assert "total_ms" in stats and "p99" in stats["total_ms"]


# -----------------------
# DRIFT / DISTRIBUTIONS
# -----------------------

def test_psi_from_dists_non_negative():
    ref_p = np.array([0.5, 0.5])
    prod_p = np.array([0.4, 0.6])
    psi = psi_from_dists(ref_p, prod_p)
    assert isinstance(psi, float)
    assert psi >= 0


def test_prod_dist_numeric_sums_to_1():
    prod_s = pd.Series([0.5, 1.5, 1.2, 2.5, 0.8])
    edges = [0, 1, 2, 3]
    labels = ["(0,1]", "(1,2]", "(2,3]"]  # labels attendus = str(pd.cut index) si tu utilises str(idx)

    labels_out, prod_p = prod_dist_numeric(prod_s, edges=edges, labels=labels)

    assert len(prod_p) == len(labels)
    assert np.isclose(float(np.sum(prod_p)), 1.0)


def test_prod_dist_numeric_empty_returns_zeros():
    prod_s = pd.Series([], dtype=float)
    edges = [0, 1, 2, 3]
    labels = ["a", "b", "c"]
    labels_out, prod_p = prod_dist_numeric(prod_s, edges=edges, labels=labels)
    assert np.all(prod_p == 0.0)


def test_prod_dist_categorical_sums_to_1():
    prod_s = pd.Series(["A", "B", "A", "B", "A"])
    labels_ref = ["A", "B"]
    labels_out, prod_p = prod_dist_categorical(prod_s, labels_ref=labels_ref)
    assert len(prod_p) == len(labels_ref)
    assert np.isclose(float(np.sum(prod_p)), 1.0)


def test_count_drift():
    df = pd.DataFrame({"feature": ["a", "b"], "psi": [0.3, 0.1]})
    assert count_drift(df, threshold=0.25) == 1


def test_compute_drift_table_basic():
    prod_inputs = pd.DataFrame(
        {"feature_num": [0.5, 1.5, 2.5], "feature_cat": ["A", "B", "A"]}
    )

    ref_rows = [
        {
            "feature": "feature_num",
            "kind": "numeric",
            "bins_json": {"edges": [0, 1, 2, 3]},
            "ref_dist_json": {"labels": ["(0,1]", "(1,2]", "(2,3]"], "p": [0.2, 0.5, 0.3]},
        },
        {
            "feature": "feature_cat",
            "kind": "categorical",
            "bins_json": None,
            "ref_dist_json": {"labels": ["A", "B"], "p": [0.7, 0.3]},
        },
    ]

    drift = compute_drift_table(prod_inputs=prod_inputs, ref_rows=ref_rows, excluded_features=set())
    assert "feature" in drift.columns
    assert "psi" in drift.columns
    assert len(drift) == 2


# -----------------------
# CONSTANTS / SECURITY / FILTERS
# -----------------------

def test_constants_sane():
    assert "endpoint" in DEFAULTS
    assert isinstance(TIME_WINDOWS, list)
    assert "ok" in PSI_THRESHOLDS and "watch" in PSI_THRESHOLDS


def test_security_excluded_features_is_set():
    assert isinstance(EXCLUDED_FEATURES, set)
    assert "SK_ID_CURR" in EXCLUDED_FEATURES


def test_apply_time_filter_all_keeps_all():
    now = pd.Timestamp.now()
    df = pd.DataFrame({"ts": [now - pd.Timedelta(days=10), now]})
    out = apply_time_filter(df, "all")
    assert len(out) == 2


def test_apply_time_filter_24h_filters():
    now = pd.Timestamp.now()
    df = pd.DataFrame({"ts": [now - pd.Timedelta(hours=1), now - pd.Timedelta(days=3)]})
    out = apply_time_filter(df, "24h")
    assert len(out) == 1