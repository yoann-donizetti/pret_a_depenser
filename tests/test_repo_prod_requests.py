from unittest.mock import Mock
import core.db.repo_prod_requests as repo_pr


def test_insert_prod_request_no_conn(monkeypatch):
    monkeypatch.setattr(repo_pr, "get_conn", lambda: None)
    repo_pr.insert_prod_request({"endpoint": "/predict"})  # ne doit pas crash


def test_insert_prod_request_executes(monkeypatch):
    fake_conn = Mock()
    monkeypatch.setattr(repo_pr, "get_conn", lambda: fake_conn)

    event = {
        "endpoint": "/predict",
        "status_code": 200,
        "latency_ms": 12.34,
        "sk_id_curr": 100001,
        "inputs": {"A": 1},
        "outputs": {"score": 1},
        "error": None,
        "message": None,
    }

    repo_pr.insert_prod_request(event)

    fake_conn.execute.assert_called_once()
    sql, params = fake_conn.execute.call_args[0]

    # SQL chargé depuis fichier => on check juste un bout stable
    assert "INSERT INTO prod_requests" in sql

    # params dict
    assert params["endpoint"] == "/predict"
    assert params["status_code"] == 200
    assert params["latency_ms"] == 12.34
    assert params["sk_id_curr"] == "100001" or params["sk_id_curr"] == 100001


def test_select_prod_requests_no_conn(monkeypatch):
    monkeypatch.setattr(repo_pr, "get_conn", lambda: None)
    out = repo_pr.select_prod_requests(endpoint="/predict", limit=10)
    assert out == []


def test_select_prod_requests_maps_rows_and_chrono_order(monkeypatch):
    fake_conn = Mock()

    # IMPORTANT: le repo fait out.reverse() pour remettre en ordre chrono
    fake_conn.execute.return_value.fetchall.return_value = [
        ("2026-01-01T10:00:01", "/predict", 200, 10.0, "100002", {"x": 2}, {"y": 2}, None, None),
        ("2026-01-01T10:00:00", "/predict", 500, 99.0, "100001", {"x": 1}, {"y": 1}, "INTERNAL_ERROR", "boom"),
    ]

    monkeypatch.setattr(repo_pr, "get_conn", lambda: fake_conn)

    out = repo_pr.select_prod_requests(endpoint="/predict", limit=2)

    assert len(out) == 2
    # après reverse(): la plus ancienne d'abord
    assert out[0]["ts"] == "2026-01-01T10:00:00"
    assert out[0]["status_code"] == 500
    assert out[0]["error"] == "INTERNAL_ERROR"
    assert out[0]["message"] == "boom"

    assert out[1]["ts"] == "2026-01-01T10:00:01"
    assert out[1]["status_code"] == 200

    fake_conn.execute.assert_called_once()