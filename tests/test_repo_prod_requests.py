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