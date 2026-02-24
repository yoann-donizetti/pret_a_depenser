# tests/test_predict_endpoint.py
import app.main as main

def _force_ready():
    main.MODEL = object()
    main.KEPT_FEATURES = ["SK_ID_CURR", "EXT_SOURCE_1"]
    main.CAT_FEATURES = []
    main.THRESHOLD = 0.5

def test_predict_ok_id_only(client, monkeypatch):
    def fake_get_features_by_id(sk_id):
        return {"SK_ID_CURR": sk_id, "EXT_SOURCE_1": 0.5}

    def fake_predict_score(model, payload, kept, cat, threshold):
        return {
            "SK_ID_CURR": payload["SK_ID_CURR"],
            "proba_default": 0.42,
            "score": 0,
            "decision": "ACCEPTED",
            "threshold": threshold,
        }

    events = []
    def fake_insert_prod_request(event):
        events.append(event)

    monkeypatch.setattr(main, "get_features_by_id", fake_get_features_by_id, raising=True)
    monkeypatch.setattr(main, "predict_score", fake_predict_score, raising=True)
    monkeypatch.setattr(main, "insert_prod_request", fake_insert_prod_request, raising=True)

    _force_ready()

    r = client.post("/predict", json={"SK_ID_CURR": 100001})
    assert r.status_code == 200
    out = r.json()
    assert out["SK_ID_CURR"] == 100001
    assert "latency_ms" in out

    assert len(events) == 1
    assert events[0]["endpoint"] == "/predict"
    assert events[0]["status_code"] == 200

def test_predict_unknown_id(client, monkeypatch):
    def fake_get_features_by_id(_sk_id):
        return None

    monkeypatch.setattr(main, "get_features_by_id", fake_get_features_by_id, raising=True)
    _force_ready()

    r = client.post("/predict", json={"SK_ID_CURR": 999999999})
    assert r.status_code in (404, 400)  # selon ton implémentation

def test_predict_rejects_extra_fields(client):
    # si schemas.py est extra="forbid"
    r = client.post("/predict", json={"SK_ID_CURR": 100001, "AMT_CREDIT": 123})
    assert r.status_code == 422