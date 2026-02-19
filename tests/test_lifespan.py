# tests/test_lifespan.py
from __future__ import annotations

import pytest
import app.main as main


@pytest.mark.anyio
async def test_lifespan_loads_artifacts(monkeypatch: pytest.MonkeyPatch):
    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat = []
    fake_thr = 0.5

    monkeypatch.setenv("BUNDLE_SOURCE", "hf")
    monkeypatch.setenv("HF_REPO_ID", "donizetti-yoann/pret-a-depenser-scoring")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    monkeypatch.setattr(
        main,
        "load_bundle_from_hf",
        lambda *a, **k: (fake_model, fake_kept, fake_cat, fake_thr),
    )

    main.MODEL = None
    main.KEPT_FEATURES = None
    main.CAT_FEATURES = None
    main.THRESHOLD = None

    app = main.create_app(enable_lifespan=True)

    async with app.router.lifespan_context(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr