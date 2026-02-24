from app.utils.errors import ApiError

def test_apierror_to_dict_without_details():
    e = ApiError(code="X", message="msg", details=None, http_status=400)
    d = e.to_dict()
    assert d == {"error": "X", "message": "msg"}

def test_apierror_to_dict_with_details():
    e = ApiError(code="X", message="msg", details={"a": 1}, http_status=422)
    d = e.to_dict()
    assert d["error"] == "X"
    assert d["message"] == "msg"
    assert d["details"] == {"a": 1}
    