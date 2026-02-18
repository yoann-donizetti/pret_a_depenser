# tests/test_lifespan.py
from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import app.main as main


def test_lifespan_loads_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # --- mocks: aucune dépendance MLflow réelle
    monkeypatch.setattr(main, "setup_mlflow", lambda *a, **k: None)
    monkeypatch.setattr(main, "load_catboost_model", lambda *a, **k: object())

    # download_run_artifacts doit retourner un dossier contenant les 3 fichiers attendus
    art_dir = tmp_path / "api_artifacts"
    art_dir.mkdir()

    (art_dir / "kept_features_top125_nocorr.txt").write_text("SK_ID_CURR\nEXT_SOURCE_1\n", encoding="utf-8")
    (art_dir / "cat_features_top125_nocorr.txt").write_text("", encoding="utf-8")
    (art_dir / "threshold_catboost_top125_nocorr.json").write_text('{"threshold": 0.5}', encoding="utf-8")

    monkeypatch.setattr(main, "download_run_artifacts", lambda artifact_uri, cache_dir: art_dir)



    app = main.create_app(enable_lifespan=True)

    # IMPORTANT: le lifespan ne s’exécute que dans le context manager
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

