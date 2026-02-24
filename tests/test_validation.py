import pytest
import math
from app.utils.validation import validate_payload
from app.utils.errors import ApiError

def test_validation_unknown_fields_rejected():
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1, "X": 999}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=True)
    assert e.value.code == "UNKNOWN_FIELDS"

def test_validation_missing_fields_raises():
    kept = ["A", "B"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=False)
    assert e.value.code == "MISSING_FIELDS"

def test_validation_bad_sk_id():
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": -1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_SK_ID_CURR"

def test_validation_numeric_type_error():
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": "not_a_number"}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_validation_ext_source_out_of_range():
    kept = ["EXT_SOURCE_1"]
    cat = []
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 2.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_validation_cat_type_error():
    kept = ["ORG"]
    cat = ["ORG"]
    payload = {"SK_ID_CURR": 1, "ORG": 123}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_validation_ok_minimal():
    kept = ["EXT_SOURCE_1", "AMT_INCOME_TOTAL"]
    cat = []
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 0.5, "AMT_INCOME_TOTAL": 1000}
    out = validate_payload(payload, kept, cat)
    assert out["EXT_SOURCE_1"] == 0.5



def test_binary_field_invalid():
    kept = ["REG_CITY_NOT_LIVE_CITY"]
    cat = []
    payload = {"SK_ID_CURR": 1, "REG_CITY_NOT_LIVE_CITY": 2}  # invalid
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_rejects_nan_and_inf():
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": float("nan")}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_VALUE"

    payload = {"SK_ID_CURR": 1, "A": float("inf")}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_VALUE"

def test_utilization_range():
    kept = ["SOME_UTILIZATION_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "SOME_UTILIZATION_FEATURE": 10.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_ratio_must_be_ge_0():
    kept = ["ANY_RATIO_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "ANY_RATIO_FEATURE": -0.1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_amt_income_must_be_gt_0():
    kept = ["AMT_INCOME_TOTAL"]
    cat = []
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_days_birth_must_be_negative_and_age_plausible():
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": 100}  # positive => invalid
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_own_car_age_range():
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 200}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"