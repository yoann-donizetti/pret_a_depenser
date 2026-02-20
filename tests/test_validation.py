# tests/test_validation.py
from __future__ import annotations

import math
import pytest

from app.utils.errors import ApiError
from app.utils.validation import validate_payload


KEPT = [
    "SK_ID_CURR",
    "EXT_SOURCE_1",
    "APARTMENTS_MODE",
    "PREV_CC_CC_UTILIZATION_MEAN_MAX",
    "BUREAU_BUREAU_DEBT_RATIO_MAX",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE",
    "FLAG_DOCUMENT_3",
    "NAME_CONTRACT_TYPE",
]
CAT = ["NAME_CONTRACT_TYPE"]


def base_payload(**overrides):
    """
    Option A: payload complet (toutes les features attendues).
    On met des valeurs "safe" par défaut, et on override pour chaque test.
    """
    p = {k: None for k in KEPT}

    # Valeurs valides par défaut
    p.update({
        "SK_ID_CURR": 1,
        "EXT_SOURCE_1": 0.2,
        "APARTMENTS_MODE": 0.3,
        "PREV_CC_CC_UTILIZATION_MEAN_MAX": 0.5,
        "BUREAU_BUREAU_DEBT_RATIO_MAX": 0.0,
        "AMT_INCOME_TOTAL": 100000,
        "AMT_CREDIT": 50000,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,
        "DAYS_BIRTH": -40 * 365.25,
        "DAYS_EMPLOYED": -2000,
        "OWN_CAR_AGE": 10,
        "FLAG_DOCUMENT_3": 1,
        "NAME_CONTRACT_TYPE": "Cash loans",
    })

    p.update(overrides)
    return p


def test_unknown_fields_rejected_when_flag_true():
    payload = base_payload(BOOM=1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "UNKNOWN_FIELDS"
    assert "BOOM" in e.value.details["unknown_fields"]


def test_unknown_fields_allowed_when_flag_false_keeps_field():
    payload = base_payload(BOOM=1)
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=False)
    assert out["BOOM"] == 1


def test_invalid_sk_id_curr_rejected():
    payload = base_payload(SK_ID_CURR=0)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "INVALID_SK_ID_CURR"


def test_numeric_int_is_accepted_for_float_field():
    payload = base_payload(EXT_SOURCE_1=1)  # int accepté
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert out["EXT_SOURCE_1"] == 1


def test_numeric_invalid_type_bool_is_rejected():
    payload = base_payload(EXT_SOURCE_1=True)  # bool refusé
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "INVALID_TYPE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_nan_rejected_like_inf():
    payload = base_payload(EXT_SOURCE_1=float("nan"))
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "INVALID_VALUE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_inf_rejected():
    payload = base_payload(EXT_SOURCE_1=float("inf"))
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "INVALID_VALUE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_out_of_range_ext_source():
    payload = base_payload(EXT_SOURCE_1=1.5)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_cat_feature_rejects_non_string():
    payload = base_payload(NAME_CONTRACT_TYPE=123)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert e.value.code == "INVALID_TYPE"
    assert e.value.details["field"] == "NAME_CONTRACT_TYPE"


def test_valid_payload_passes_all_range_checks():
    payload = base_payload()
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert out["SK_ID_CURR"] == 1