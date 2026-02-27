# tests/test_logging.py
import app.main as main

def _force_ready():
    main.MODEL = object()
    main.KEPT_FEATURES = ["SK_ID_CURR", "EXT_SOURCE_1"]
    main.CAT_FEATURES = []
    main.THRESHOLD = 0.5

def test_db_logging_failure_does_not_break(client, monkeypatch):
    def fake_get_features_by_id(sk_id):
        return {"SK_ID_CURR": sk_id, "EXT_SOURCE_1": 0.5}

    def fake_predict_score(
        model,
        payload,
        kept=None,
        cat=None,
        threshold=None,
        *,
        kept_features=None,
        cat_features=None,
        thread_count=None,
        **kwargs,
    ):
        # strict: on garde l'opti
        assert thread_count == 1

        # compat: ton code peut passer kept_features/cat_features en keyword
        if kept_features is not None:
            kept = kept_features
        if cat_features is not None:
            cat = cat_features

        return {
            "SK_ID_CURR": payload["SK_ID_CURR"],
            "proba_default": 0.42,
            "score": 0,
            "decision": "ACCEPTED",
            "threshold": threshold,
        }

    def boom(_event):
        raise RuntimeError("db down")

    monkeypatch.setattr(main, "get_features_by_id", fake_get_features_by_id, raising=True)
    monkeypatch.setattr(main, "predict_score", fake_predict_score, raising=True)
    monkeypatch.setattr(main, "insert_prod_request", boom, raising=True)

    _force_ready()

    r = client.post("/predict", json={"SK_ID_CURR": 100001})
    assert r.status_code == 200