# tests/test_predict.py

import numpy as np
import pandas as pd
import pytest

from app.model.predict import build_X, _extract_proba_class1, predict_score


# -------------------------
# Fakes models
# -------------------------
class FakeModelProba:
    def __init__(self, out):
        self._out = out

    def predict_proba(self, X):
        return self._out


class FakeModelPredict:
    def __init__(self, out):
        self._out = out

    def predict(self, X):
        return self._out


# -------------------------
# build_X
# -------------------------
def test_build_X_numeric_none_becomes_nan_and_cat_none_becomes_missing_str():
    kept = ["NUM1", "CAT1", "NUM2"]
    cat = ["CAT1"]

    payload = {
        "NUM1": None,         # doit devenir NaN
        "CAT1": None,         # doit devenir "__MISSING__"
        # NUM2 absent => None -> NaN
    }

    X = build_X(payload, kept, cat)

    assert list(X.columns) == kept
    assert X.shape == (1, 3)

    assert pd.isna(X.loc[0, "NUM1"])
    assert pd.isna(X.loc[0, "NUM2"])
    assert X.loc[0, "CAT1"] == "__MISSING__"
    assert isinstance(X.loc[0, "CAT1"], str)


def test_build_X_cat_is_cast_to_str_even_if_number():
    kept = ["CAT1"]
    cat = ["CAT1"]
    payload = {"CAT1": 123}

    X = build_X(payload, kept, cat)

    assert X.loc[0, "CAT1"] == "123"
    assert isinstance(X.loc[0, "CAT1"], str)


# -------------------------
# _extract_proba_class1
# -------------------------
def test_extract_proba_empty_raises():
    with pytest.raises(ValueError):
        _extract_proba_class1([])


@pytest.mark.parametrize(
    "pred, expected",
    [
        (0.42, 0.42),                 # float
        ([0.42], 0.42),               # [p]
        ([[0.42]], 0.42),             # [[p]]
        ([[0.1, 0.9]], 0.9),          # [[p0, p1]]
        (np.array([[0.2, 0.8]]), 0.8),
    ],
)
def test_extract_proba_various_formats(pred, expected):
    assert _extract_proba_class1(pred) == pytest.approx(expected)


# -------------------------
# predict_score
# -------------------------
def test_predict_score_uses_predict_proba_when_available_and_thresholding():
    kept = ["SK_ID_CURR", "X1"]
    cat = []
    payload = {"SK_ID_CURR": 100001, "X1": 1.0}

    # proba classe 1 = 0.7
    model = FakeModelProba([[0.3, 0.7]])

    out = predict_score(model, payload, kept, cat, threshold=0.65)

    assert out["SK_ID_CURR"] == 100001
    assert out["proba_default"] == pytest.approx(0.7, rel=0, abs=1e-6)
    assert out["score"] == 1
    assert out["decision"] == "REFUSED"
    assert out["threshold"] == 0.65


def test_predict_score_falls_back_to_predict_when_no_predict_proba():
    kept = ["SK_ID_CURR", "X1"]
    cat = []
    payload = {"SK_ID_CURR": 100001, "X1": 1.0}

    # predict renvoie direct une "proba" scalaire
    model = FakeModelPredict(0.12)

    out = predict_score(model, payload, kept, cat, threshold=0.5)

    assert out["score"] == 0
    assert out["decision"] == "ACCEPTED"
    assert out["proba_default"] == pytest.approx(0.12, rel=0, abs=1e-6)


def test_predict_score_predict_output_as_class_still_works_degenerate_proba():
    kept = ["SK_ID_CURR", "X1"]
    cat = []
    payload = {"SK_ID_CURR": 100001, "X1": 1.0}

    # modèle renvoie une classe 0/1 (degenerate proba)
    model = FakeModelPredict(1)

    out = predict_score(model, payload, kept, cat, threshold=0.5)

    assert out["proba_default"] == pytest.approx(1.0, rel=0, abs=1e-6)
    assert out["score"] == 1
    assert out["decision"] == "REFUSED"