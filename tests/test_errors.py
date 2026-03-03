
"""
Tests unitaires pour la classe ApiError (gestion des erreurs API personnalisées).
Vérifie la conversion en dictionnaire avec ou sans détails.
"""
from app.utils.errors import ApiError

def test_apierror_to_dict_without_details():
    """
    Vérifie que to_dict() retourne un dictionnaire sans clé 'details' si aucun détail n'est fourni.
    """
    e = ApiError(code="X", message="msg", details=None, http_status=400)
    d = e.to_dict()
    assert d == {"error": "X", "message": "msg"}

def test_apierror_to_dict_with_details():
    """
    Vérifie que to_dict() inclut la clé 'details' si des détails sont fournis.
    """
    e = ApiError(code="X", message="msg", details={"a": 1}, http_status=422)
    d = e.to_dict()
    assert d["error"] == "X"
    assert d["message"] == "msg"
    assert d["details"] == {"a": 1}
    