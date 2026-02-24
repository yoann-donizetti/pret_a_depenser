import pytest
from unittest.mock import Mock

import core.db.repo_features_store as repo_fs





def test_get_features_by_id_returns_none_if_no_conn(monkeypatch):
    monkeypatch.setattr(repo_fs, "get_conn", lambda: None)
    assert repo_fs.get_features_by_id(100001) is None




def test_get_features_by_id_returns_none_if_not_found(monkeypatch):
    fake_conn = Mock()
    fake_conn.execute.return_value.fetchone.return_value = None

    monkeypatch.setattr(repo_fs, "get_conn", lambda: fake_conn)

    result = repo_fs.get_features_by_id(100001)

    assert result is None
    fake_conn.execute.assert_called_once()




def test_get_features_by_id_returns_data(monkeypatch):
    fake_data = {"SK_ID_CURR": 100001, "EXT_SOURCE_1": 0.5}

    fake_conn = Mock()
    fake_conn.execute.return_value.fetchone.return_value = (fake_data,)

    monkeypatch.setattr(repo_fs, "get_conn", lambda: fake_conn)

    result = repo_fs.get_features_by_id(100001)

    assert result == fake_data
    fake_conn.execute.assert_called_once()


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
    assert "INSERT INTO prod_requests" in sql
    # params est un dict (selon ton repo.py)
    assert params["endpoint"] == "/predict"
    assert params["status_code"] == 200