# tests/test_integration.py

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.model.loader as loader_mod


class FakeModel:
    def predict_proba(self, X):
        # 1 ligne -> proba défaut = 0.8
        return [[0.2, 0.8]]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_model = FakeModel()

    # Bundle minimal cohérent avec validate_payload + predict_score
    fake_kept = ["SK_ID_CURR", "EXT_SOURCE_1"]
    fake_cat: list[str] = []
    fake_thr = 0.5

    # Force mode local, et on évite HF
    monkeypatch.setenv("BUNDLE_SOURCE", "local")
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # IMPORTANT: ne pas toucher à une vraie DB en CI
    monkeypatch.setattr(main, "init_db", lambda: None)

    def fake_loader(**_kwargs):
        return (fake_model, fake_kept, fake_cat, fake_thr)

    # Patch les 2 cibles possibles (selon comment le code importe)
    monkeypatch.setattr(main, "load_bundle_from_local", fake_loader)
    monkeypatch.setattr(loader_mod, "load_bundle_from_local", fake_loader)

    app = main.create_app(enable_lifespan=True)

    with TestClient(app) as c:
        yield c


def test_health_is_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_predict_success(client: TestClient) -> None:
    payload: Dict[str, Any] = {
        "SK_ID_CURR": 123,
        "EXT_SOURCE_1": 0.42,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()

    # Contrat minimal
    expected_keys = {"SK_ID_CURR", "proba_default", "score", "decision", "threshold", "latency_ms"}
    assert expected_keys.issubset(data.keys())

    assert data["SK_ID_CURR"] == 123
    assert isinstance(data["proba_default"], float)
    assert 0.0 <= data["proba_default"] <= 1.0

    assert data["score"] in (0, 1)
    assert data["decision"] in ("ACCEPTED", "REFUSED")
    assert isinstance(data["threshold"], (int, float))
    assert isinstance(data["latency_ms"], (int, float))


def test_predict_rejects_unknown_field(client: TestClient) -> None:
    payload: Dict[str, Any] = {
        "SK_ID_CURR": 123,
        "EXT_SOURCE_1": 0.42,
        "HACK": 1,  # champ inconnu -> doit être rejeté si reject_unknown_fields=True
    }

    r = client.post("/predict", json=payload)
    assert r.status_code in (400, 422), r.text


def test_predict_rejects_out_of_range(client: TestClient) -> None:
    payload: Dict[str, Any] = {
        "SK_ID_CURR": 123,
        "EXT_SOURCE_1": 1.42,  # hors [0,1] => doit être rejeté
    }

    r = client.post("/predict", json=payload)
    assert r.status_code in (400, 422), r.text