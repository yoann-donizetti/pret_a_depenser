# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas import PredictRequest

def test_predict_request_accepts_only_id():
    req = PredictRequest(SK_ID_CURR=100001)
    assert req.SK_ID_CURR == 100001

def test_predict_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PredictRequest(SK_ID_CURR=100001, AMT_CREDIT=123)