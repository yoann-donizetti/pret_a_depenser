import pandas as pd

from monitoring.lib.data import load_prod_data


def test_load_prod_data_empty(monkeypatch):
    # mock DB read to return no logs
    monkeypatch.setattr("monitoring.lib.data.select_prod_requests", lambda endpoint, limit: [])

    meta, inputs, outputs, rows = load_prod_data(
        endpoint="/predict",
        limit=None,
        time_window="all",
        excluded_features=set(),
    )

    assert meta.empty
    assert inputs.empty
    assert outputs.empty
    assert rows == []


def test_load_prod_data_ok(monkeypatch):
    fake_rows = [
        {
            "ts": pd.Timestamp.now(),
            "endpoint": "/predict",
            "status_code": 200,
            "latency_ms": 123.0,
            "inputs": {"A": 1, "SK_ID_CURR": 100001},
            "outputs": {"decision": "ACCEPTED"},
            "error": None,
            "message": None,
        }
    ]

    monkeypatch.setattr("monitoring.lib.data.select_prod_requests", lambda endpoint, limit: fake_rows)

    meta, inputs, outputs, rows = load_prod_data(
        endpoint="/predict",
        limit=100,
        time_window="all",
        excluded_features={"SK_ID_CURR"},
    )

    assert not meta.empty
    assert "SK_ID_CURR" not in inputs.columns
    assert "decision" in outputs.columns
    assert len(rows) == 1