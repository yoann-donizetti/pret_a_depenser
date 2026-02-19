from __future__ import annotations

from pathlib import Path
import json
import pytest

import app.model.loader as loader


def _write_minimal_artifacts(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    kept = base / "kept_features_top125_nocorr.txt"
    cat = base / "cat_features_top125_nocorr.txt"
    thr = base / "threshold_catboost_top125_nocorr.json"

    kept.write_text("SK_ID_CURR\nEXT_SOURCE_1\n", encoding="utf-8")
    cat.write_text("", encoding="utf-8")
    thr.write_text(json.dumps({"threshold": 0.42}), encoding="utf-8")

    return {"kept": kept, "cat": cat, "thr": thr}


def test_load_bundle_from_local_loads_all_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    art_dir = tmp_path / "api_artifacts"
    model_dir.mkdir()

    files = _write_minimal_artifacts(art_dir)
    model_file = model_dir / "model.cb"
    model_file.write_bytes(b"FAKE_CATBOOST_MODEL")

    fake_model = object()
    monkeypatch.setattr(loader, "_load_catboost_from_file", lambda *_a, **_k: fake_model)

    model, kept, cat, thr = loader.load_bundle_from_local(
        model_path=model_file,
        kept_path=files["kept"],
        cat_path=files["cat"],
        threshold_path=files["thr"],
    )

    assert model is fake_model
    assert kept == ["SK_ID_CURR", "EXT_SOURCE_1"]
    assert cat == []
    assert thr == 0.42


def test_load_bundle_from_hf_downloads_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    model_dir = repo_root / "model"
    art_dir = repo_root / "api_artifacts"
    model_dir.mkdir(parents=True)

    files = _write_minimal_artifacts(art_dir)
    model_path = model_dir / "model.cb"
    model_path.write_bytes(b"FAKE")

    fake_model = object()
    monkeypatch.setattr(loader, "_load_catboost_from_file", lambda *_a, **_k: fake_model)

    def fake_hf_hub_download(repo_id: str, filename: str, token=None, **kwargs):
        local = repo_root / filename
        assert local.exists(), f"Fichier demandé inexistant: {local}"
        return str(local)

    monkeypatch.setattr(loader, "hf_hub_download", fake_hf_hub_download)

    model, kept, cat, thr = loader.load_bundle_from_hf(
        repo_id="donizetti-yoann/pret-a-depenser-scoring",
        model_path="model/model.cb",
        kept_path="api_artifacts/kept_features_top125_nocorr.txt",
        cat_path="api_artifacts/cat_features_top125_nocorr.txt",
        threshold_path="api_artifacts/threshold_catboost_top125_nocorr.json",
        token="fake",
    )

    assert model is fake_model
    assert kept == ["SK_ID_CURR", "EXT_SOURCE_1"]
    assert cat == []
    assert thr == 0.42