
"""
Tests unitaires pour la fonction validate_payload (validation des données d'entrée API).
Vérifie la gestion des champs inconnus, manquants, des types, des valeurs extrêmes et des règles métier.
"""
import pytest
import math
from app.utils.validation import validate_payload
from app.utils.errors import ApiError

def test_validation_unknown_fields_rejected():
    """
    Vérifie que la présence de champs inconnus dans le payload lève une ApiError avec le code 'UNKNOWN_FIELDS'.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1, "X": 999}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=True)
    assert e.value.code == "UNKNOWN_FIELDS"

def test_validation_missing_fields_raises():
    """
    Vérifie que l'absence de champs obligatoires lève une ApiError avec le code 'MISSING_FIELDS'.
    """
    kept = ["A", "B"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=False)
    assert e.value.code == "MISSING_FIELDS"

def test_validation_bad_sk_id():
    """
    Vérifie que SK_ID_CURR négatif lève une ApiError avec le code 'INVALID_SK_ID_CURR'.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": -1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_SK_ID_CURR"

def test_validation_numeric_type_error():
    """
    Vérifie qu'une erreur de type sur un champ numérique lève une ApiError avec le code 'INVALID_TYPE'.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": "not_a_number"}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_validation_ext_source_out_of_range():
    """
    Vérifie qu'une valeur hors bornes pour EXT_SOURCE_1 lève une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["EXT_SOURCE_1"]
    cat = []
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 2.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_validation_cat_type_error():
    """
    Vérifie qu'une erreur de type sur un champ catégoriel lève une ApiError avec le code 'INVALID_TYPE'.
    """
    kept = ["ORG"]
    cat = ["ORG"]
    payload = {"SK_ID_CURR": 1, "ORG": 123}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_validation_ok_minimal():
    """
    Vérifie qu'un payload minimal valide passe la validation sans erreur.
    """
    kept = ["EXT_SOURCE_1", "AMT_INCOME_TOTAL"]
    cat = []
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 0.5, "AMT_INCOME_TOTAL": 1000}
    out = validate_payload(payload, kept, cat)
    assert out["EXT_SOURCE_1"] == 0.5



def test_binary_field_invalid():
    """
    Vérifie qu'une valeur non binaire pour un champ binaire lève une ApiError avec le code 'INVALID_TYPE'.
    """
    kept = ["REG_CITY_NOT_LIVE_CITY"]
    cat = []
    payload = {"SK_ID_CURR": 1, "REG_CITY_NOT_LIVE_CITY": 2}  # invalid
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_rejects_nan_and_inf():
    """
    Vérifie que la présence de NaN ou d'infini dans un champ lève une ApiError avec le code 'INVALID_VALUE'.
    """
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
    """
    Vérifie qu'une valeur hors bornes pour une feature d'utilisation lève une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["SOME_UTILIZATION_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "SOME_UTILIZATION_FEATURE": 10.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_ratio_must_be_ge_0():
    """
    Vérifie qu'un ratio négatif lève une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["ANY_RATIO_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "ANY_RATIO_FEATURE": -0.1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_amt_income_must_be_gt_0():
    """
    Vérifie qu'un montant de revenu nul ou négatif lève une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["AMT_INCOME_TOTAL"]
    cat = []
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_days_birth_must_be_negative_and_age_plausible():
    """
    Vérifie que DAYS_BIRTH doit être négatif et plausible, sinon lève une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": 100}  # positive => invalid
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_own_car_age_range():
    """
    Vérifie qu'une valeur trop élevée pour OWN_CAR_AGE lève une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 200}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"