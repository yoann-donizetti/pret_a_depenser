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

def test_upsert_features_no_conn(monkeypatch):
    monkeypatch.setattr(repo_fs, "get_conn", lambda: None)
    repo_fs.upsert_features(100001, {"A": 1})  # ne doit pas crash


def test_upsert_features_executes(monkeypatch):
    fake_conn = Mock()
    monkeypatch.setattr(repo_fs, "get_conn", lambda: fake_conn)

    repo_fs.upsert_features(100001, {"A": 1})

    fake_conn.execute.assert_called_once()
    sql, params = fake_conn.execute.call_args[0]
    assert "sk_id_curr" in params
    assert int(params["sk_id_curr"]) == 100001
    assert "data" in params