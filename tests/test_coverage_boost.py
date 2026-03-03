
"""
Tests complémentaires pour augmenter la couverture du code sur des branches spécifiques.
Vérifie les comportements de fallback, de gestion d'erreur et d'appel de modèle.
"""
import inspect
import pytest

import app.main as mainmod
import app.model.predict as predmod


def test_bundle_source_auto_fallbacks(monkeypatch):
    """
    Teste le fallback automatique de la fonction _bundle_source selon la config.
    - Si BUNDLE_SOURCE est invalide et HF_REPO_ID présent, retourne 'hf'.
    - Si BUNDLE_SOURCE est invalide et HF_REPO_ID absent, retourne 'local'.
    """
    # Cas 1: BUNDLE_SOURCE invalide + HF_REPO_ID présent => "hf"
    monkeypatch.setattr(mainmod.config, "BUNDLE_SOURCE", "weird", raising=False)
    monkeypatch.setattr(mainmod.config, "HF_REPO_ID", "some/repo", raising=False)
    assert mainmod._bundle_source() == "hf"

    # Cas 2: BUNDLE_SOURCE invalide + HF_REPO_ID absent => "local"
    monkeypatch.setattr(mainmod.config, "HF_REPO_ID", None, raising=False)
    assert mainmod._bundle_source() == "local"


def test_safe_log_ne_crashe_pas(monkeypatch):
    """
    Vérifie que _safe_log ne fait jamais planter l'API même si insert_prod_request lève une exception.
    """
    def boom(_event):
        raise RuntimeError("db down")

    monkeypatch.setattr(mainmod, "insert_prod_request", boom, raising=True)

    # Doit swallow l'exception
    mainmod._safe_log({"endpoint": "/x"})


def test_call_model_thread_count_branch():
    """
    Teste la fonction _call_model pour les cas où la méthode accepte ou non le paramètre thread_count.
    """
    class M:
        # accepte thread_count
        def f(self, X, thread_count=None):
            return ("ok", thread_count)

    out = predmod._call_model(M().f, X=[["x"]], thread_count=1)
    assert out == ("ok", 1)

    class M2:
        # n'accepte PAS thread_count
        def f(self, X):
            return ("ok2",)

    out2 = predmod._call_model(M2().f, X=[["x"]], thread_count=1)
    assert out2 == ("ok2",)