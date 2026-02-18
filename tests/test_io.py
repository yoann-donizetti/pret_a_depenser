# tests/test_io.py

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.io import parse_json, load_txt_list
from app.utils.errors import ApiError


def test_parse_json_empty():
    with pytest.raises(ApiError) as e:
        parse_json("   ")
    assert e.value.code == "EMPTY_JSON"


def test_parse_json_invalid():
    with pytest.raises(ApiError) as e:
        parse_json("{bad json}")
    assert e.value.code == "INVALID_JSON"
    assert "line" in e.value.details
    assert "col" in e.value.details


def test_parse_json_not_a_dict():
    with pytest.raises(ApiError) as e:
        parse_json('["a", "b"]')
    assert e.value.code == "INVALID_PAYLOAD"


def test_load_txt_list_strips_and_ignores_empty(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("  a  \n\n b\n   \n", encoding="utf-8")

    out = load_txt_list(p)
    assert out == ["a", "b"]