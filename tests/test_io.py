import json
import pytest

from app.utils.io import parse_json
from app.utils.errors import ApiError

def test_parse_json_empty_raises():
    with pytest.raises(ApiError) as e:
        parse_json("   ")
    assert e.value.code == "EMPTY_JSON"

def test_parse_json_invalid_raises():
    with pytest.raises(ApiError) as e:
        parse_json("{bad json")
    assert e.value.code == "INVALID_JSON"
    assert "line" in (e.value.details or {})

def test_parse_json_not_object_raises():
    with pytest.raises(ApiError) as e:
        parse_json(json.dumps([1, 2, 3]))
    assert e.value.code == "INVALID_PAYLOAD"

def test_parse_json_ok():
    obj = parse_json('{"threshold": 0.42}')
    assert obj["threshold"] == 0.42