# tests/test_loader.py

from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace

import app.model.loader as loader


def test_setup_mlflow_sets_tracking_uri_and_env(monkeypatch):
    called = {"uri": None}

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: called.__setitem__("uri", uri)
    )

    monkeypatch.setattr(loader, "_get_mlflow", lambda: fake_mlflow)

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    loader.setup_mlflow("sqlite:///mlflow.db")

    assert called["uri"] == "sqlite:///mlflow.db"
    assert os.environ["MLFLOW_TRACKING_URI"] == "sqlite:///mlflow.db"
    assert "MLFLOW_ARTIFACT_URI" not in os.environ


def test_setup_mlflow_with_artifact_root_sets_env_and_creates_dir(monkeypatch, tmp_path: Path):
    called = {"uri": None}

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: called.__setitem__("uri", uri)
    )

    monkeypatch.setattr(loader, "_get_mlflow", lambda: fake_mlflow)

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    artifact_root = tmp_path / "mlflow_artifacts"
    assert not artifact_root.exists()

    loader.setup_mlflow("http://mlflow:5000", artifact_root=artifact_root)

    assert called["uri"] == "http://mlflow:5000"
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert artifact_root.exists()
    assert artifact_root.is_dir()
    assert os.environ["MLFLOW_ARTIFACT_URI"] == artifact_root.resolve().as_uri()


def test_download_run_artifacts_creates_dst_and_returns_path(monkeypatch, tmp_path: Path):
    returned = tmp_path / "downloaded" / "api_artifacts"

    def fake_download_artifacts(artifact_uri: str, dst_path: str):
        assert artifact_uri == "runs:/123/api_artifacts"
        assert Path(dst_path).exists()  # dst_dir créé avant appel
        return str(returned)

    fake_mlflow = SimpleNamespace(
        artifacts=SimpleNamespace(download_artifacts=fake_download_artifacts)
    )

    monkeypatch.setattr(loader, "_get_mlflow", lambda: fake_mlflow)

    dst_dir = tmp_path / "cache"
    assert not dst_dir.exists()

    out = loader.download_run_artifacts("runs:/123/api_artifacts", dst_dir)

    assert dst_dir.exists()
    assert out == returned


def test_load_catboost_model_calls_mlflow_catboost_load_model(monkeypatch):
    called = {"uri": None}
    fake_model = object()

    def fake_load_model(uri: str):
        called["uri"] = uri
        return fake_model

    fake_mlflow = SimpleNamespace(
        catboost=SimpleNamespace(load_model=fake_load_model)
    )

    monkeypatch.setattr(loader, "_get_mlflow", lambda: fake_mlflow)

    model = loader.load_catboost_model("models:/home_credit_catboost/Production")

    assert called["uri"] == "models:/home_credit_catboost/Production"
    assert model is fake_model