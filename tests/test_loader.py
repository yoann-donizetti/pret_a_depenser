from pathlib import Path
import json
import pytest

import app.model.loader as loader

class DummyCat:
    def __init__(self):
        self.loaded = None
    def load_model(self, path):
        self.loaded = path

def test_load_bundle_from_local_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CatBoostClassifier", DummyCat)

    with pytest.raises(FileNotFoundError):
        loader.load_bundle_from_local(
            model_path=tmp_path / "model.cb",
            kept_path=tmp_path / "kept.txt",
            cat_path=tmp_path / "cat.txt",
            threshold_path=tmp_path / "thr.json",
        )

def test_load_bundle_from_local_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CatBoostClassifier", DummyCat)

    model_file = tmp_path / "model.cb"
    kept_file = tmp_path / "kept.txt"
    cat_file = tmp_path / "cat.txt"
    thr_file = tmp_path / "thr.json"

    model_file.write_text("fake", encoding="utf-8")
    kept_file.write_text("A\nB\n", encoding="utf-8")
    cat_file.write_text("B\n", encoding="utf-8")
    thr_file.write_text(json.dumps({"threshold": 0.33}), encoding="utf-8")

    model, kept, cat, thr = loader.load_bundle_from_local(
        model_path=model_file,
        kept_path=kept_file,
        cat_path=cat_file,
        threshold_path=thr_file,
    )

    assert kept == ["A", "B"]
    assert cat == ["B"]
    assert thr == 0.33
    assert model.loaded == str(model_file)

def test_load_bundle_from_hf_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CatBoostClassifier", DummyCat)

    # Simule les fichiers téléchargés HF => on renvoie des paths tmp
    model_file = tmp_path / "model.cb"
    kept_file = tmp_path / "kept.txt"
    cat_file = tmp_path / "cat.txt"
    thr_file = tmp_path / "thr.json"

    model_file.write_text("fake", encoding="utf-8")
    kept_file.write_text("A\nB\n", encoding="utf-8")
    cat_file.write_text("B\n", encoding="utf-8")
    thr_file.write_text('{"threshold": 0.9}', encoding="utf-8")

    def fake_hf_download(repo_id, filename, token=None):
        mapping = {
            "model/model.cb": model_file,
            "api_artifacts/kept.txt": kept_file,
            "api_artifacts/cat.txt": cat_file,
            "api_artifacts/thr.json": thr_file,
        }
        return str(mapping[filename])

    monkeypatch.setattr(loader, "hf_hub_download", fake_hf_download)

    model, kept, cat, thr = loader.load_bundle_from_hf(
        repo_id="x/y",
        model_path="model/model.cb",
        kept_path="api_artifacts/kept.txt",
        cat_path="api_artifacts/cat.txt",
        threshold_path="api_artifacts/thr.json",
        token="t",
    )

    assert thr == 0.9
    assert kept == ["A", "B"]
    assert cat == ["B"]