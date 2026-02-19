import math
import pytest

from app.utils.errors import ApiError
from app.utils.validation import validate_payload


# IMPORTANT :
# - kept_features doit contenir tous les champs qu'on teste, sinon on tombe en UNKNOWN_FIELDS
# - validate_payload ne "complète" pas les champs manquants : il renvoie le payload tel quel
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
    "FLAG_DOCUMENT_3",       # binaire (dans BINARY_FIELDS)
    "NAME_CONTRACT_TYPE",    # catégorielle (dans CAT)
]

CAT = ["NAME_CONTRACT_TYPE"]


# -----------------------------
# Unknown fields behavior
# -----------------------------
def test_unknown_fields_rejected_when_flag_true():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 0.2, "BOOM": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "UNKNOWN_FIELDS"
    assert "BOOM" in e.value.details["unknown_fields"]


def test_unknown_fields_allowed_when_flag_false_keeps_field():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 0.2, "BOOM": 1}
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=False)

    assert out["SK_ID_CURR"] == 1
    assert out["EXT_SOURCE_1"] == 0.2
    # ton validate_payload ne filtre pas, donc BOOM reste
    assert out["BOOM"] == 1


# -----------------------------
# SK_ID_CURR behavior (optionnel mais si présent => int > 0)
# -----------------------------
def test_missing_sk_id_curr_allowed():
    payload = {"EXT_SOURCE_1": 0.2}
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert out["EXT_SOURCE_1"] == 0.2
    assert "SK_ID_CURR" not in out


def test_invalid_sk_id_curr_rejected():
    payload = {"SK_ID_CURR": 0, "EXT_SOURCE_1": 0.2}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "INVALID_SK_ID_CURR"


# -----------------------------
# Missing optional fields (pas de fill)
# -----------------------------
def test_missing_optional_fields_are_not_filled_with_none():
    payload = {"SK_ID_CURR": 1}
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    # validate_payload renvoie le payload tel quel
    assert out == payload
    assert "EXT_SOURCE_1" not in out
    assert "DAYS_BIRTH" not in out


# -----------------------------
# Type checking numeric (bool exclu) + finite checks
# -----------------------------
def test_numeric_int_is_accepted_for_float_field():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 1}  # int accepté (number)
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert out["EXT_SOURCE_1"] == 1


def test_numeric_invalid_type_bool_is_rejected():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": True}  # bool exclu des numériques
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "INVALID_TYPE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_nan_rejected_like_inf():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": float("nan")}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "INVALID_VALUE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_inf_rejected():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": float("inf")}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "INVALID_VALUE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


# -----------------------------
# Range checks
# -----------------------------
def test_out_of_range_ext_source():
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 1.5}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "EXT_SOURCE_1"


def test_out_of_range_mode_suffix():
    payload = {"SK_ID_CURR": 1, "APARTMENTS_MODE": -0.1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "APARTMENTS_MODE"


def test_out_of_range_utilization_too_high():
    payload = {"SK_ID_CURR": 1, "PREV_CC_CC_UTILIZATION_MEAN_MAX": 10.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "PREV_CC_CC_UTILIZATION_MEAN_MAX"


def test_out_of_range_utilization_negative_too_low():
    payload = {"SK_ID_CURR": 1, "PREV_CC_CC_UTILIZATION_MEAN_MAX": -0.5}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "PREV_CC_CC_UTILIZATION_MEAN_MAX"


def test_out_of_range_ratio_negative():
    payload = {"SK_ID_CURR": 1, "BUREAU_BUREAU_DEBT_RATIO_MAX": -1.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "BUREAU_BUREAU_DEBT_RATIO_MAX"


def test_amt_income_total_must_be_positive():
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "AMT_INCOME_TOTAL"


def test_amt_credit_must_be_non_negative():
    payload = {"SK_ID_CURR": 1, "AMT_CREDIT": -10}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "AMT_CREDIT"


def test_cnt_fields_must_be_non_negative():
    payload = {"SK_ID_CURR": 1, "DEF_30_CNT_SOCIAL_CIRCLE": -1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "DEF_30_CNT_SOCIAL_CIRCLE"


def test_days_birth_must_be_negative():
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": 100}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "DAYS_BIRTH"


def test_days_birth_age_years_too_high():
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": -200 * 365.25}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "DAYS_BIRTH"
    assert "age_years" in e.value.details


def test_days_generic_bounds():
    payload = {"SK_ID_CURR": 1, "DAYS_EMPLOYED": 200000}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "DAYS_EMPLOYED"


def test_own_car_age_bounds():
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 150}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "OUT_OF_RANGE"
    assert e.value.details["field"] == "OWN_CAR_AGE"


# -----------------------------
# Binary + categorical types
# -----------------------------
def test_binary_field_rejects_string():
    payload = {"SK_ID_CURR": 1, "FLAG_DOCUMENT_3": "yes"}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "INVALID_TYPE"
    assert e.value.details["field"] == "FLAG_DOCUMENT_3"


def test_cat_feature_rejects_non_string():
    payload = {"SK_ID_CURR": 1, "NAME_CONTRACT_TYPE": 123}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)

    assert e.value.code == "INVALID_TYPE"
    assert e.value.details["field"] == "NAME_CONTRACT_TYPE"


def test_cat_feature_accepts_string():
    payload = {"SK_ID_CURR": 1, "NAME_CONTRACT_TYPE": "Cash loans"}
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert out["NAME_CONTRACT_TYPE"] == "Cash loans"



def test_valid_payload_passes_all_range_checks():
    payload = {
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
    }
    out = validate_payload(payload, KEPT, CAT, reject_unknown_fields=True)
    assert out["SK_ID_CURR"] == 1