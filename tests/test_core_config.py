import importlib
import os


def _reload_core_config_with_env(monkeypatch, env: dict):
    # on remplace complètement l'environnement visible par le module
    monkeypatch.setattr(os, "environ", env.copy(), raising=True)

    import core.config as config_mod
    return importlib.reload(config_mod)


def test_config_exports_project_root(monkeypatch):
    c = _reload_core_config_with_env(monkeypatch, {})
    assert hasattr(c, "PROJECT_ROOT")
    assert str(c.PROJECT_ROOT)


def test_config_reads_database_url(monkeypatch):
    c = _reload_core_config_with_env(monkeypatch, {"DATABASE_URL": "postgresql://x"})
    assert getattr(c, "DATABASE_URL", None) == "postgresql://x"


def test_config_empty_database_url(monkeypatch):
    c = _reload_core_config_with_env(monkeypatch, {})
    assert getattr(c, "DATABASE_URL", None) in (None, "")