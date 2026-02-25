from unittest.mock import Mock
import core.db.repo_ref_dist as repo


def test_load_all_ref_no_conn(monkeypatch):
    monkeypatch.setattr(repo, "get_conn", lambda: None)
    assert repo.load_all_ref() == []


def test_load_one_ref_no_conn(monkeypatch):
    monkeypatch.setattr(repo, "get_conn", lambda: None)
    assert repo.load_one_ref("AMT_CREDIT") is None


def test_load_all_ref_returns_rows(monkeypatch):
    fake_conn = Mock()

    fake_conn.execute.return_value.fetchall.return_value = [
        (
            "AMT_CREDIT",
            "numeric",
            {"edges": [0, 1]},
            {"labels": ["bin1"], "p": [1.0]},
            100,
            "2026-01-01",
        )
    ]

    monkeypatch.setattr(repo, "get_conn", lambda: fake_conn)

    result = repo.load_all_ref()

    assert len(result) == 1
    assert result[0]["feature"] == "AMT_CREDIT"
    assert result[0]["kind"] == "numeric"
    assert result[0]["n_ref"] == 100


def test_load_one_ref_returns_dict(monkeypatch):
    fake_conn = Mock()

    fake_conn.execute.return_value.fetchone.return_value = (
        "AMT_CREDIT",
        "numeric",
        {"edges": [0, 1]},
        {"labels": ["bin1"], "p": [1.0]},
        100,
        "2026-01-01",
    )

    monkeypatch.setattr(repo, "get_conn", lambda: fake_conn)

    result = repo.load_one_ref("AMT_CREDIT")

    assert result["feature"] == "AMT_CREDIT"
    assert result["kind"] == "numeric"
    assert result["n_ref"] == 100


def test_upsert_executes_sql(monkeypatch):
    fake_conn = Mock()
    monkeypatch.setattr(repo, "get_conn", lambda: fake_conn)

    repo.upsert_ref_feature_dist(
        feature="AMT_CREDIT",
        kind="numeric",
        bins_json={"edges": [0, 1]},
        ref_dist_json={"labels": ["bin1"], "p": [1.0]},
        n_ref=100,
    )

    fake_conn.execute.assert_called_once()