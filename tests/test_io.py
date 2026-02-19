from __future__ import annotations

import importlib
import pytest

import app.config as config
import app.main as main


@pytest.mark.anyio
async def test_lifespan_loads_artifacts_hf(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUNDLE_SOURCE", "hf")
    monkeypatch.setenv("HF_REPO_ID", "donizetti-yoann/pret-a-depenser-scoring")

    # reload config + main pour prendre les env
    importlib.reload(config)
    importlib.reload(main)

    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat = []
    fake_thr = 0.5

    monkeypatch.setattr(main, "load_bundle_from_hf", lambda **_k: (fake_model, fake_kept, fake_cat, fake_thr))

    app = main.create_app(enable_lifespan=True)

    async with main.lifespan(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr


@pytest.mark.anyio
async def test_lifespan_loads_artifacts_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUNDLE_SOURCE", "local")

    importlib.reload(config)
    importlib.reload(main)

    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat = []
    fake_thr = 0.5

    monkeypatch.setattr(main, "load_bundle_from_local", lambda **_k: (fake_model, fake_kept, fake_cat, fake_thr))

    app = main.create_app(enable_lifespan=True)

    async with main.lifespan(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr