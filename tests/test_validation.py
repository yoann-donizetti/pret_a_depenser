
"""
Tests unitaires pour la fonction validate_payload (validation des données d'entrée API).

Ce module vérifie la robustesse de la validation des données utilisateur pour l'API, en testant :
- la gestion des champs inconnus et manquants
- la validation des types (numérique, binaire, catégoriel)
- le respect des bornes et des règles métier spécifiques (bornes, ratios, dates, etc.)
- la gestion des cas limites et des valeurs extrêmes
- la levée d'ApiError avec le bon code et les bons détails en cas d'erreur

Chaque fonction de test cible un aspect précis de la validation pour garantir une couverture maximale et la fiabilité du code de validation.
"""
import pytest
import math
from app.utils.validation import validate_payload
from app.utils.errors import ApiError

def test_validation_unknown_fields_rejected():
    """
    Teste le rejet des champs inconnus dans le payload.
    Doit lever une ApiError avec le code 'UNKNOWN_FIELDS'.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1, "X": 999}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=True)
    assert e.value.code == "UNKNOWN_FIELDS"

def test_validation_missing_fields_raises():
    """
    Teste la détection des champs obligatoires manquants.
    Doit lever une ApiError avec le code 'MISSING_FIELDS'.
    """
    kept = ["A", "B"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=False)
    assert e.value.code == "MISSING_FIELDS"

def test_validation_bad_sk_id():
    """
    Teste la validation de SK_ID_CURR négatif.
    Doit lever une ApiError avec le code 'INVALID_SK_ID_CURR'.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": -1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_SK_ID_CURR"

def test_validation_numeric_type_error():
    """
    Teste la détection d'un type incorrect pour un champ numérique.
    Doit lever une ApiError avec le code 'INVALID_TYPE'.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": "not_a_number"}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_validation_ext_source_out_of_range():
    """
    Teste la détection d'une valeur hors bornes pour EXT_SOURCE_1.
    Doit lever une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["EXT_SOURCE_1"]
    cat = []
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 2.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_validation_cat_type_error():
    """
    Teste la détection d'un type incorrect pour un champ catégoriel.
    Doit lever une ApiError avec le code 'INVALID_TYPE'.
    """
    kept = ["ORG"]
    cat = ["ORG"]
    payload = {"SK_ID_CURR": 1, "ORG": 123}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_validation_ok_minimal():
    """
    Teste qu'un payload minimal et valide passe la validation sans erreur.
    """
    kept = ["EXT_SOURCE_1", "AMT_INCOME_TOTAL"]
    cat = []
    payload = {"SK_ID_CURR": 1, "EXT_SOURCE_1": 0.5, "AMT_INCOME_TOTAL": 1000}
    out = validate_payload(payload, kept, cat)
    assert out["EXT_SOURCE_1"] == 0.5



def test_binary_field_invalid():
    """
    Teste le rejet d'une valeur non binaire pour un champ binaire.
    Doit lever une ApiError avec le code 'INVALID_TYPE'.
    """
    kept = ["REG_CITY_NOT_LIVE_CITY"]
    cat = []
    payload = {"SK_ID_CURR": 1, "REG_CITY_NOT_LIVE_CITY": 2}  # invalid
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"

def test_rejects_nan_and_inf():
    """
    Teste le rejet des valeurs NaN et infini dans un champ numérique.
    Doit lever une ApiError avec le code 'INVALID_VALUE'.
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
    Teste la détection d'une valeur hors bornes pour une feature d'utilisation.
    Doit lever une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["SOME_UTILIZATION_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "SOME_UTILIZATION_FEATURE": 10.0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_ratio_must_be_ge_0():
    """
    Teste la détection d'un ratio négatif.
    Doit lever une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["ANY_RATIO_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "ANY_RATIO_FEATURE": -0.1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_amt_income_must_be_gt_0():
    """
    Teste la détection d'un montant de revenu nul ou négatif.
    Doit lever une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["AMT_INCOME_TOTAL"]
    cat = []
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 0}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_days_birth_must_be_negative_and_age_plausible():
    """
    Teste la validation de DAYS_BIRTH (doit être négatif et plausible).
    Doit lever une ApiError avec le code 'OUT_OF_RANGE' si la valeur est incohérente.
    """
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": 100}  # positive => invalid
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_own_car_age_range():
    """
    Teste la détection d'une valeur trop élevée pour OWN_CAR_AGE.
    Doit lever une ApiError avec le code 'OUT_OF_RANGE'.
    """
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 200}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"



def ok_payload(**kwargs):
    """
    Génère un payload de test standard avec SK_ID_CURR=1 et les champs supplémentaires passés en argument.
    """
    p = {"SK_ID_CURR": 1}
    p.update(kwargs)
    return p




def test_unknown_fields_allowed_when_flag_false():
    """
    Vérifie que les champs inconnus sont acceptés si reject_unknown_fields=False.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1, "X": 999}
    out = validate_payload(payload, kept, cat, reject_unknown_fields=False)
    assert out["X"] == 999


def test_missing_fields_message_contains_count():
    """
    Vérifie que le message d'erreur pour champs manquants contient bien le nombre de champs absents.
    """
    kept = ["A", "B", "C"]
    cat = []
    payload = {"SK_ID_CURR": 1, "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat, reject_unknown_fields=False)
    assert e.value.code == "MISSING_FIELDS"
    assert e.value.details["count"] == 2



def test_sk_id_curr_none_is_accepted():
    """
    Vérifie que SK_ID_CURR à None est accepté sans erreur.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": None, "A": 1}
    out = validate_payload(payload, kept, cat)
    assert out["SK_ID_CURR"] is None


def test_sk_id_curr_bad_type():
    """
    Vérifie qu'un SK_ID_CURR de type incorrect lève une ApiError.
    """
    kept = ["A"]
    cat = []
    payload = {"SK_ID_CURR": "1", "A": 1}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_SK_ID_CURR"


def test_binary_field_accepts_bool_true():
    """
    Vérifie qu'un champ binaire accepte la valeur True.
    """
    kept = ["REG_CITY_NOT_LIVE_CITY"]
    cat = []
    payload = ok_payload(REG_CITY_NOT_LIVE_CITY=True)
    out = validate_payload(payload, kept, cat)
    assert out["REG_CITY_NOT_LIVE_CITY"] is True


def test_binary_field_accepts_0_and_1():
    """
    Vérifie qu'un champ binaire accepte les valeurs 0 et 1.
    """
    kept = ["FLAG_DOCUMENT_3"]
    cat = []
    out0 = validate_payload(ok_payload(FLAG_DOCUMENT_3=0), kept, cat)
    out1 = validate_payload(ok_payload(FLAG_DOCUMENT_3=1), kept, cat)
    assert out0["FLAG_DOCUMENT_3"] == 0
    assert out1["FLAG_DOCUMENT_3"] == 1



def test_rejects_minus_inf():
    """
    Vérifie que -inf est rejeté pour un champ numérique.
    """
    kept = ["A"]
    cat = []
    payload = ok_payload(A=float("-inf"))
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_VALUE"


def test_bool_is_not_a_number_for_numeric_feature():
    """
    Vérifie qu'un booléen n'est pas accepté comme nombre pour un champ numérique.
    """
    kept = ["A"]
    cat = []
    payload = ok_payload(A=True)  # bool should be rejected as numeric
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "INVALID_TYPE"



def test_categorical_accepts_str():
    """
    Vérifie qu'un champ catégoriel accepte une chaîne de caractères.
    """
    kept = ["ORG"]
    cat = ["ORG"]
    payload = ok_payload(ORG="Business Entity Type 3")
    out = validate_payload(payload, kept, cat)
    assert out["ORG"].startswith("Business")


def test_ext_source_lower_bound_ok():
    """
    Vérifie que la borne inférieure (0.0) d'un EXT_SOURCE est acceptée.
    """
    kept = ["EXT_SOURCE_2"]
    cat = []
    payload = ok_payload(EXT_SOURCE_2=0.0)
    out = validate_payload(payload, kept, cat)
    assert out["EXT_SOURCE_2"] == 0.0


def test_mode_out_of_range():
    """
    Vérifie qu'une valeur hors bornes pour un champ _MODE lève une ApiError.
    """
    kept = ["SOME_MODE"]
    cat = []
    payload = ok_payload(SOME_MODE=1.5)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"



def test_utilization_upper_bound_ok():
    """
    Vérifie que la borne supérieure (3.0) d'un champ UTILIZATION est acceptée.
    """
    kept = ["ANY_UTILIZATION_FEATURE"]
    cat = []
    payload = ok_payload(ANY_UTILIZATION_FEATURE=3.0)
    out = validate_payload(payload, kept, cat)
    assert out["ANY_UTILIZATION_FEATURE"] == 3.0


def test_utilization_too_negative():
    """
    Vérifie qu'une valeur trop négative pour un champ UTILIZATION lève une ApiError.
    """
    kept = ["ANY_UTILIZATION_FEATURE"]
    cat = []
    payload = ok_payload(ANY_UTILIZATION_FEATURE=-0.5)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"



def test_amt_goods_price_negative_rejected():
    """
    Vérifie qu'une valeur négative pour un champ AMT_ (hors AMT_INCOME_TOTAL) lève une ApiError.
    """
    kept = ["AMT_GOODS_PRICE"]
    cat = []
    payload = ok_payload(AMT_GOODS_PRICE=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"


def test_cnt_feature_negative_rejected_prefix():
    """
    Vérifie qu'une valeur négative pour un champ préfixé CNT_ lève une ApiError.
    """
    kept = ["CNT_CHILDREN"]
    cat = []
    payload = ok_payload(CNT_CHILDREN=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"


def test_cnt_feature_negative_rejected_suffix():
    """
    Vérifie qu'une valeur négative pour un champ suffixé _COUNT lève une ApiError.
    """
    kept = ["SOME_COUNT"]
    cat = []
    payload = ok_payload(SOME_COUNT=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"


def test_ratio_feature_ge_0_ok():
    """
    Vérifie qu'une valeur nulle pour un champ RATIO est acceptée.
    """
    kept = ["MY_RATIO_FEATURE"]
    cat = []
    payload = ok_payload(MY_RATIO_FEATURE=0.0)
    out = validate_payload(payload, kept, cat)
    assert out["MY_RATIO_FEATURE"] == 0.0



def test_days_generic_out_of_bounds():
    """
    Vérifie qu'une valeur hors bornes pour un champ DAYS_ lève une ApiError.
    """
    kept = ["DAYS_EMPLOYED"]
    cat = []
    payload = ok_payload(DAYS_EMPLOYED=200000)  # out of allowed [-100000, 100000]
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"


def test_days_non_birth_positive_allowed_if_in_bounds():
    """
    Vérifie qu'une valeur positive pour un champ DAYS_ non birth est acceptée si dans les bornes.
    """
    kept = ["DAYS_ID_PUBLISH"]
    cat = []
    payload = ok_payload(DAYS_ID_PUBLISH=10)  # allowed by your generic bound
    out = validate_payload(payload, kept, cat)
    assert out["DAYS_ID_PUBLISH"] == 10


def test_days_birth_age_too_high():
    """
    Vérifie qu'un âge incohérent (trop élevé) pour DAYS_BIRTH lève une ApiError.
    """
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = ok_payload(DAYS_BIRTH=-200000)  # ~547 years -> should fail
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"



def test_own_car_age_negative_rejected():
    """
    Vérifie qu'une valeur négative pour OWN_CAR_AGE lève une ApiError.
    """
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = ok_payload(OWN_CAR_AGE=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"



def ok_payload(**kwargs):
    p = {"SK_ID_CURR": 1}
    p.update(kwargs)
    return p

def test_categorical_binary_field_is_ignored_in_cat_loop():
    """
    Vérifie que les champs binaires sont ignorés dans la boucle des catégorielles.
    """
    kept = ["FLAG_DOCUMENT_3"]
    cat = ["FLAG_DOCUMENT_3"]  # volontairement "mauvais" pour forcer le chemin
    payload = ok_payload(FLAG_DOCUMENT_3=1)
    out = validate_payload(payload, kept, cat)
    assert out["FLAG_DOCUMENT_3"] == 1

def test_numeric_none_is_skipped():
    """
    Vérifie qu'une valeur None pour un champ numérique est ignorée (pas d'erreur).
    """
    kept = ["A"]
    cat = []
    payload = ok_payload(A=None)
    out = validate_payload(payload, kept, cat)
    assert out["A"] is None

def test_categorical_none_is_skipped():
    """
    Vérifie qu'une valeur None pour un champ catégoriel est ignorée (pas d'erreur).
    """
 
    kept = ["ORG"]
    cat = ["ORG"]
    payload = ok_payload(ORG=None)
    out = validate_payload(payload, kept, cat)
    assert out["ORG"] is None

def test_amt_other_than_income_negative_rejected():
    """
    Vérifie qu'une valeur négative pour un champ AMT_ (hors income) lève une ApiError.
    """
    kept = ["AMT_GOODS_PRICE"]
    cat = []
    payload = ok_payload(AMT_GOODS_PRICE=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_count_suffix_cnt_negative_rejected():
    """
    Vérifie qu'une valeur négative pour un champ suffixé _CNT lève une ApiError.
    """
    kept = ["FOO_CNT"]
    cat = []
    payload = ok_payload(FOO_CNT=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_count_suffix_count_negative_rejected():
    """
    Vérifie qu'une valeur négative pour un champ suffixé _COUNT lève une ApiError.
    """
    kept = ["FOO_COUNT"]
    cat = []
    payload = ok_payload(FOO_COUNT=-1)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_days_generic_out_of_bounds_rejected():
    """
    Vérifie qu'une valeur hors bornes pour un champ DAYS_ lève une ApiError (autre test de branche).
    """
    kept = ["DAYS_EMPLOYED"]
    cat = []
    payload = ok_payload(DAYS_EMPLOYED=200000)
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_mode_in_range_ok():
    """
    Vérifie qu'une valeur dans [0,1] pour un champ _MODE est acceptée.
    """
    kept = ["SOME_MODE"]
    cat = []
    payload = ok_payload(SOME_MODE=0.5)
    out = validate_payload(payload, kept, cat)
    assert out["SOME_MODE"] == 0.5

def test_days_birth_age_too_high_raises():
    """
    Vérifie qu'un âge > 120 ans pour DAYS_BIRTH lève une ApiError.
    """
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": -int(130 * 365.25)}
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"

def test_count_fields_non_negative_ok():
    """
    Vérifie que des champs de type count (CNT_, _CNT, _COUNT) >= 0 sont acceptés.
    """
    kept = ["FOO_CNT", "FOO_COUNT", "CNT_BAR"]
    cat = []
    payload = {"SK_ID_CURR": 1, "FOO_CNT": 0, "FOO_COUNT": 2, "CNT_BAR": 1}
    out = validate_payload(payload, kept, cat)
    assert out["FOO_CNT"] == 0


def test_own_car_age_in_range_ok():
    """
    Vérifie qu'une valeur dans [0,100] pour OWN_CAR_AGE est acceptée.
    """
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 5}
    out = validate_payload(payload, kept, cat)
    assert out["OWN_CAR_AGE"] == 5

def test_utilization_in_range_ok():
    """
    Vérifie qu'une valeur dans [0,3] pour une feature UTILIZATION est acceptée.
    """
    kept = ["SOME_UTILIZATION_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "SOME_UTILIZATION_FEATURE": 1.0}
    out = validate_payload(payload, kept, cat)
    assert out["SOME_UTILIZATION_FEATURE"] == 1.0


def test_ratio_ge_0_ok():
    """
    Vérifie qu'une valeur nulle pour un champ RATIO est acceptée.
    """
    kept = ["ANY_RATIO_FEATURE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "ANY_RATIO_FEATURE": 0.0}
    out = validate_payload(payload, kept, cat)
    assert out["ANY_RATIO_FEATURE"] == 0.0



def test_amt_feature_non_negative_ok():
    """
    Vérifie qu'une valeur nulle pour un champ AMT_ (hors income) est acceptée.

    Couvre le chemin où k.startswith("AMT_") et f >= 0 (donc PAS d'erreur).
    Utile si ton coverage pointe une branche dans ce bloc.
    """
    kept = ["AMT_GOODS_PRICE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "AMT_GOODS_PRICE": 0}
    out = validate_payload(payload, kept, cat)
    assert out["AMT_GOODS_PRICE"] == 0


def test_cnt_fields_non_negative_ok():
    """
    Vérifie que des champs de type count (CNT_, _CNT, _COUNT) >= 0 sont acceptés.

    Couvre le chemin:
    if "CNT_" in k or k.endswith("_CNT") or k.endswith("_COUNT"):
        if f < 0: raise
    Ici on met f >= 0 pour traverser la branche sans lever.
    """
    kept = ["CNT_FOO", "BAR_CNT", "BAZ_COUNT"]
    cat = []
    payload = {"SK_ID_CURR": 1, "CNT_FOO": 0, "BAR_CNT": 2, "BAZ_COUNT": 1}
    out = validate_payload(payload, kept, cat)
    assert out["CNT_FOO"] == 0
    assert out["BAR_CNT"] == 2
    assert out["BAZ_COUNT"] == 1


def test_days_birth_age_incoherent_too_high_raises():
    """
    Vérifie qu'un âge incohérent (>120 ans) pour DAYS_BIRTH lève une ApiError.

    Couvre le chemin:
    age_years = abs(f)/365.25
    if not (0 < age_years < 120): raise ApiError(...)
    On force un âge > 120 ans.
    """
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": -int(130 * 365.25)}  # ~130 ans
    with pytest.raises(ApiError) as e:
        validate_payload(payload, kept, cat)
    assert e.value.code == "OUT_OF_RANGE"


def test_own_car_age_in_range_ok():
    """
    Vérifie qu'une valeur dans [0,100] pour OWN_CAR_AGE est acceptée (autre test de branche).

    Couvre le chemin:
    if k == "OWN_CAR_AGE" and (f < 0 or f > 100): raise
    Ici on met une valeur OK pour traverser sans lever.
    """
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 10}
    out = validate_payload(payload, kept, cat)
    assert out["OWN_CAR_AGE"] == 10

import pytest
from app.utils.validation import validate_payload
from app.utils.errors import ApiError


def test_days_birth_age_plausible_does_not_raise():
    """
    Vérifie qu'un âge plausible pour DAYS_BIRTH ne lève pas d'erreur.

    Couvre la branche FALSE de:
    if not (0 < age_years < 120)
    -> âge plausible, donc pas d'erreur
    """
    kept = ["DAYS_BIRTH"]
    cat = []
    payload = {"SK_ID_CURR": 1, "DAYS_BIRTH": -int(30 * 365.25)}  # ~30 ans

    out = validate_payload(payload, kept, cat)
    assert out["DAYS_BIRTH"] == payload["DAYS_BIRTH"]


def test_own_car_age_valid_does_not_raise():
    """
    Vérifie qu'une valeur valide pour OWN_CAR_AGE ne lève pas d'erreur.
    Couvre la branche FALSE de:
    if k == "OWN_CAR_AGE" and (f < 0 or f > 100)
    -> valeur OK, donc pas d'erreur
    """
    kept = ["OWN_CAR_AGE"]
    cat = []
    payload = {"SK_ID_CURR": 1, "OWN_CAR_AGE": 5}

    out = validate_payload(payload, kept, cat)
    assert out["OWN_CAR_AGE"] == 5