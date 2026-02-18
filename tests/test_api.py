# tests/test_api.py
from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.utils.errors import ApiError


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # App de test : lifespan OFF (pas de MLflow, pas de download)
    app = main.create_app(enable_lifespan=False)

    # Etat "ready" fake
    main.MODEL = object()
    main.KEPT_FEATURES = ["SK_ID_CURR", "EXT_SOURCE_1"]
    main.CAT_FEATURES = []
    main.THRESHOLD = 0.5

    return TestClient(app)


def test_health_not_ready(client: TestClient):
    main.MODEL = None
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "not_ready"


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_success(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    payload_in: Dict[str, Any] = {"SK_ID_CURR": 123, "EXT_SOURCE_1": 0.2}

    def _fake_validate(payload, kept, cat, reject_unknown_fields=True):
        return payload

    def _fake_predict_score(model, payload, kept, cat, threshold):
        return {
            "SK_ID_CURR": payload.get("SK_ID_CURR"),
            "proba_default": 0.42,
            "score": 0,
            "decision": "ACCEPTED",
            "threshold": float(threshold),
        }

    monkeypatch.setattr(main, "validate_payload", _fake_validate)
    monkeypatch.setattr(main, "predict_score", _fake_predict_score)

    r = client.post("/predict", json=payload_in)
    assert r.status_code == 200
    data = r.json()

    assert data["SK_ID_CURR"] == 123
    assert "latency_ms" in data
    assert data["decision"] in ("ACCEPTED", "REFUSED")


def test_predict_api_error_returns_http_status(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    payload_in = {"SK_ID_CURR": 0, "EXT_SOURCE_1": 0.2}

    def _fake_validate(*args, **kwargs):
        raise ApiError(
            code="INVALID_SK_ID_CURR",
            message="SK_ID_CURR doit être un entier > 0.",
            details={"SK_ID_CURR": 0},
            http_status=400,
        )

    monkeypatch.setattr(main, "validate_payload", _fake_validate)

    r = client.post("/predict", json=payload_in)
    assert r.status_code == 400
    data = r.json()
    assert data["error"] == "INVALID_SK_ID_CURR"
    assert "latency_ms" in data


def test_predict_internal_error_returns_500(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    payload_in = {"SK_ID_CURR": 123, "EXT_SOURCE_1": 0.2}

    def _fake_validate(payload, *args, **kwargs):
        return payload

    def _fake_predict_score(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "validate_payload", _fake_validate)
    monkeypatch.setattr(main, "predict_score", _fake_predict_score)

    r = client.post("/predict", json=payload_in)
    assert r.status_code == 500
    data = r.json()
    assert data["error"] == "INTERNAL_ERROR"
    assert "boom" in data["message"]
    assert "latency_ms" in data