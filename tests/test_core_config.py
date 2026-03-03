
"""
Tests unitaires pour le module core.config :
Vérifie la gestion des variables d'environnement et des constantes de configuration.
"""
import importlib
import os


def _reload_core_config_with_env(monkeypatch, env: dict):
    """
    Recharge le module core.config avec un environnement simulé.
    Paramètres :
        monkeypatch : fixture pytest pour patcher l'environnement.
        env (dict) : Dictionnaire des variables d'environnement à utiliser.
    Retour :
        Module core.config rechargé avec le nouvel environnement.
    """
    # on remplace complètement l'environnement visible par le module
    monkeypatch.setattr(os, "environ", env.copy(), raising=True)

    import core.config as config_mod
    return importlib.reload(config_mod)


def test_config_exports_project_root(monkeypatch):
    """
    Vérifie que la constante PROJECT_ROOT est bien exportée par core.config.
    """
    c = _reload_core_config_with_env(monkeypatch, {})
    assert hasattr(c, "PROJECT_ROOT")
    assert str(c.PROJECT_ROOT)


def test_config_reads_database_url(monkeypatch):
    """
    Vérifie que la variable d'environnement DATABASE_URL est bien lue par core.config.
    """
    c = _reload_core_config_with_env(monkeypatch, {"DATABASE_URL": "postgresql://x"})
    assert getattr(c, "DATABASE_URL", None) == "postgresql://x"


def test_config_empty_database_url(monkeypatch):
    """
    Vérifie le comportement de core.config quand DATABASE_URL est absente ou vide.
    """
    c = _reload_core_config_with_env(monkeypatch, {})
    assert getattr(c, "DATABASE_URL", None) in (None, "")