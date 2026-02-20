from __future__ import annotations

import importlib
import pytest

import app.main as main
import app.config as config


@pytest.mark.anyio
async def test_lifespan_loads_artifacts_hf(monkeypatch: pytest.MonkeyPatch):
    # Arrange
    monkeypatch.setenv("BUNDLE_SOURCE", "hf")
    monkeypatch.setenv("HF_REPO_ID", "donizetti-yoann/pret-a-depenser-scoring")

    # IMPORTANT: ne pas toucher DB en CI
    monkeypatch.setattr(main, "init_db", lambda: None)

    importlib.reload(config)
    importlib.reload(main)

    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat: list[str] = []
    fake_thr = 0.5

    monkeypatch.setattr(
        main,
        "load_bundle_from_hf",
        lambda **_k: (fake_model, fake_kept, fake_cat, fake_thr),
    )

    app = main.create_app(enable_lifespan=True)

    # Act + Assert
    async with main.lifespan(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr


@pytest.mark.anyio
async def test_lifespan_loads_artifacts_local(monkeypatch: pytest.MonkeyPatch):
    # Arrange
    monkeypatch.setenv("BUNDLE_SOURCE", "local")

    # IMPORTANT: ne pas toucher DB en CI
    monkeypatch.setattr(main, "init_db", lambda: None)

    importlib.reload(config)
    importlib.reload(main)

    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat: list[str] = []
    fake_thr = 0.5

    monkeypatch.setattr(
        main,
        "load_bundle_from_local",
        lambda **_k: (fake_model, fake_kept, fake_cat, fake_thr),
    )

    app = main.create_app(enable_lifespan=True)

    # Act + Assert
    async with main.lifespan(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr