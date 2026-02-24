import numpy as np
import pandas as pd

from app.model.predict import build_X, predict_score

class DummyProbaModel:
    def predict_proba(self, X):
        # 1 ligne, 2 colonnes => p1 = 0.7
        return np.array([[0.3, 0.7]])

class DummyScalarModel:
    def predict(self, X):
        return np.array([0.2])

def test_build_X_shapes_and_missing():
    payload = {"A": 1, "B": None, "C": None}
    kept = ["A", "B", "C"]
    cat = ["C"]
    X = build_X(payload, kept, cat)

    assert list(X.columns) == kept
    assert X.shape == (1, 3)
    assert pd.isna(X.loc[0, "B"])
    assert X.loc[0, "C"] == "__MISSING__"

def test_predict_score_uses_predict_proba():
    payload = {"SK_ID_CURR": 1, "A": 0}
    out = predict_score(DummyProbaModel(), payload, kept_features=["A"], cat_features=[], threshold=0.5)
    assert out["proba_default"] == 0.7
    assert out["score"] == 1
    assert out["decision"] == "REFUSED"

def test_predict_score_fallback_predict():
    payload = {"SK_ID_CURR": 1, "A": 0}
    out = predict_score(DummyScalarModel(), payload, kept_features=["A"], cat_features=[], threshold=0.5)
    assert out["proba_default"] == 0.2
    assert out["score"] == 0
    assert out["decision"] == "ACCEPTED"