import numpy as np

from app.model.predict import build_row, predict_score


class DummyProbaModel:
    def predict_proba(self, X):
        # X est une liste 2D : [[...]]
        return np.array([[0.3, 0.7]])


class DummyScalarModel:
    def predict(self, X):
        return np.array([0.2])


def test_build_row_order_and_missing():
    payload = {"A": 1, "B": None, "C": None}
    kept = ["A", "B", "C"]
    cat = ["C"]

    row, cat_idx = build_row(payload, kept, cat)

    # ordre respecté
    assert row[0] == 1

    # None -> np.nan pour non cat
    assert np.isnan(row[1])

    # None -> "__MISSING__" pour cat
    assert row[2] == "__MISSING__"

    # index catégoriel correct
    assert cat_idx == [2]


def test_predict_score_uses_predict_proba():
    payload = {"SK_ID_CURR": 1, "A": 0}

    out = predict_score(
        DummyProbaModel(),
        payload,
        kept_features=["A"],
        cat_features=[],
        threshold=0.5,
    )

    assert out["proba_default"] == 0.7
    assert out["score"] == 1
    assert out["decision"] == "REFUSED"


def test_predict_score_fallback_predict():
    payload = {"SK_ID_CURR": 1, "A": 0}

    out = predict_score(
        DummyScalarModel(),
        payload,
        kept_features=["A"],
        cat_features=[],
        threshold=0.5,
    )

    assert out["proba_default"] == 0.2
    assert out["score"] == 0
    assert out["decision"] == "ACCEPTED"