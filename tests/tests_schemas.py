
"""
Tests unitaires pour le schéma PredictRequest (validation Pydantic).
Vérifie l'acceptation du champ SK_ID_CURR et le rejet des champs supplémentaires.
"""
# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas import PredictRequest

def test_predict_request_accepts_only_id():
    """
    Vérifie que PredictRequest accepte uniquement le champ SK_ID_CURR.
    """
    req = PredictRequest(SK_ID_CURR=100001)
    assert req.SK_ID_CURR == 100001

def test_predict_request_rejects_extra_fields():
    """
    Vérifie que PredictRequest rejette les champs supplémentaires non attendus.
    """
    with pytest.raises(ValidationError):
        PredictRequest(SK_ID_CURR=100001, AMT_CREDIT=123)