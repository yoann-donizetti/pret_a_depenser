from __future__ import annotations

from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.model.loader as loader_mod


class FakeModel:
    def predict_proba(self, X):
        return [[0.2, 0.8]]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    fake_model = FakeModel()

    # On teste l’intégration sur un bundle MINIMAL
    fake_kept = ["SK_ID_CURR", "EXT_SOURCE_1"]
    fake_cat: list[str] = []
    fake_thr = 0.5

    monkeypatch.setenv("BUNDLE_SOURCE", "local")
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fake_loader(**kwargs):
        return (fake_model, fake_kept, fake_cat, fake_thr)

    # Patch robuste (les 2 cibles)
    monkeypatch.setattr(main, "load_bundle_from_local", fake_loader)
    monkeypatch.setattr(loader_mod, "load_bundle_from_local", fake_loader)

    app = main.create_app(enable_lifespan=True)
    with TestClient(app) as test_client:
        yield test_client


def test_integration_health_and_predict(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "not_ready")  # selon timing, mais en général "ok"

    payload: Dict[str, Any] = {
        "SK_ID_CURR": 123,
        "EXT_SOURCE_1": 0.42,  # borne OK [0,1]
    }

    r = client.post("/predict", json=payload)
    if r.status_code != 200:
        print("ERROR:", r.status_code, r.json())

    assert r.status_code == 200

    data = r.json()
    expected_keys = {"SK_ID_CURR", "proba_default", "score", "decision", "threshold", "latency_ms"}
    assert expected_keys.issubset(set(data.keys()))
    assert data["SK_ID_CURR"] == 123
    assert isinstance(data["proba_default"], float)
    assert data["score"] in (0, 1)
    assert data["decision"] in ("ACCEPTED", "REFUSED")
    assert isinstance(data["threshold"], (int, float))
    assert isinstance(data["latency_ms"], (int, float))