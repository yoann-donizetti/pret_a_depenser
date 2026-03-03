
"""
Tests unitaires pour la fonction parse_json (gestion de la désérialisation JSON et des erreurs associées).
Vérifie la robustesse face aux entrées vides, invalides ou non conformes.
"""
import json
import pytest

from app.utils.io import parse_json
from app.utils.errors import ApiError

def test_parse_json_empty_raises():
    """
    Vérifie que parse_json lève une ApiError avec le code 'EMPTY_JSON' si la chaîne est vide ou ne contient que des espaces.
    """
    with pytest.raises(ApiError) as e:
        parse_json("   ")
    assert e.value.code == "EMPTY_JSON"

def test_parse_json_invalid_raises():
    """
    Vérifie que parse_json lève une ApiError avec le code 'INVALID_JSON' si la chaîne n'est pas un JSON valide.
    """
    with pytest.raises(ApiError) as e:
        parse_json("{bad json")
    assert e.value.code == "INVALID_JSON"
    assert "line" in (e.value.details or {})

def test_parse_json_not_object_raises():
    """
    Vérifie que parse_json lève une ApiError avec le code 'INVALID_PAYLOAD' si le JSON n'est pas un objet (ex : liste).
    """
    with pytest.raises(ApiError) as e:
        parse_json(json.dumps([1, 2, 3]))
    assert e.value.code == "INVALID_PAYLOAD"

def test_parse_json_ok():
    """
    Vérifie que parse_json retourne bien le dictionnaire attendu pour un JSON valide.
    """
    obj = parse_json('{"threshold": 0.42}')
    assert obj["threshold"] == 0.42