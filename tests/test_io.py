from __future__ import annotations

import importlib
import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from app.utils.io import load_txt_list, parse_json
from app.utils.errors import ApiError

import app.config as config
import app.main as main


@pytest.mark.anyio
async def test_lifespan_loads_artifacts_hf(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUNDLE_SOURCE", "hf")
    monkeypatch.setenv("HF_REPO_ID", "donizetti-yoann/pret-a-depenser-scoring")

    # reload config + main pour prendre les env
    importlib.reload(config)
    importlib.reload(main)

    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat = []
    fake_thr = 0.5

    monkeypatch.setattr(main, "load_bundle_from_hf", lambda **_k: (fake_model, fake_kept, fake_cat, fake_thr))

    app = main.create_app(enable_lifespan=True)

    async with main.lifespan(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr


@pytest.mark.anyio
async def test_lifespan_loads_artifacts_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUNDLE_SOURCE", "local")

    importlib.reload(config)
    importlib.reload(main)

    fake_model = object()
    fake_kept = ["SK_ID_CURR"]
    fake_cat = []
    fake_thr = 0.5

    monkeypatch.setattr(main, "load_bundle_from_local", lambda **_k: (fake_model, fake_kept, fake_cat, fake_thr))

    app = main.create_app(enable_lifespan=True)

    async with main.lifespan(app):
        assert main.MODEL is fake_model
        assert main.KEPT_FEATURES == fake_kept
        assert main.CAT_FEATURES == fake_cat
        assert main.THRESHOLD == fake_thr
        class TestLoadTxtList:
            def test_load_txt_list_basic(self):
                with TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "test.txt"
                    path.write_text("item1\nitem2\nitem3\n", encoding="utf-8")
                    result = load_txt_list(path)
                    assert result == ["item1", "item2", "item3"]

            def test_load_txt_list_with_whitespace(self):
                with TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "test.txt"
                    path.write_text("  item1  \nitem2\n  item3\n", encoding="utf-8")
                    result = load_txt_list(path)
                    assert result == ["item1", "item2", "item3"]

            def test_load_txt_list_with_empty_lines(self):
                with TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "test.txt"
                    path.write_text("item1\n\nitem2\n\n\nitem3\n", encoding="utf-8")
                    result = load_txt_list(path)
                    assert result == ["item1", "item2", "item3"]

            def test_load_txt_list_empty_file(self):
                with TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "test.txt"
                    path.write_text("", encoding="utf-8")
                    result = load_txt_list(path)
                    assert result == []


        class TestParseJson:
            def test_parse_json_valid_dict(self):
                result = parse_json('{"key": "value", "number": 42}')
                assert result == {"key": "value", "number": 42}

            def test_parse_json_empty_dict(self):
                result = parse_json('{}')
                assert result == {}

            def test_parse_json_nested_dict(self):
                result = parse_json('{"outer": {"inner": "value"}}')
                assert result == {"outer": {"inner": "value"}}

            def test_parse_json_empty_string(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json("")
                assert exc_info.value.code == "EMPTY_JSON"

            def test_parse_json_whitespace_only(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json("   \n  \t  ")
                assert exc_info.value.code == "EMPTY_JSON"

            def test_parse_json_none(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json(None)
                assert exc_info.value.code == "EMPTY_JSON"

            def test_parse_json_invalid_syntax(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json('{"key": invalid}')
                assert exc_info.value.code == "INVALID_JSON"
                assert "line" in exc_info.value.details
                assert "col" in exc_info.value.details

            def test_parse_json_list_instead_of_dict(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json('[1, 2, 3]')
                assert exc_info.value.code == "INVALID_PAYLOAD"

            def test_parse_json_string_instead_of_dict(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json('"just a string"')
                assert exc_info.value.code == "INVALID_PAYLOAD"

            def test_parse_json_number_instead_of_dict(self):
                with pytest.raises(ApiError) as exc_info:
                    parse_json('42')
                assert exc_info.value.code == "INVALID_PAYLOAD"
                def test_parse_json_true_instead_of_dict(self):
                    with pytest.raises(ApiError) as exc_info:
                        parse_json('true')
                    assert exc_info.value.code == "INVALID_PAYLOAD"

                def test_parse_json_false_instead_of_dict(self):
                    with pytest.raises(ApiError) as exc_info:
                        parse_json('false')
                    assert exc_info.value.code == "INVALID_PAYLOAD"

                def test_parse_json_null_instead_of_dict(self):
                    with pytest.raises(ApiError) as exc_info:
                        parse_json('null')
                    assert exc_info.value.code == "INVALID_PAYLOAD"

                def test_parse_json_malformed_bracket(self):
                    with pytest.raises(ApiError) as exc_info:
                        parse_json('{"key": "value"')
                    assert exc_info.value.code == "INVALID_JSON"

                def test_parse_json_trailing_comma(self):
                    with pytest.raises(ApiError) as exc_info:
                        parse_json('{"key": "value",}')
                    assert exc_info.value.code == "INVALID_JSON"

                def test_parse_json_single_quotes(self):
                    with pytest.raises(ApiError) as exc_info:
                        parse_json("{'key': 'value'}")
                    assert exc_info.value.code == "INVALID_JSON"
                    def test_parse_json_true_instead_of_dict():
                        with pytest.raises(ApiError) as exc_info:
                            parse_json('true')
                        assert exc_info.value.code == "INVALID_PAYLOAD"

                    def test_parse_json_false_instead_of_dict():
                        with pytest.raises(ApiError) as exc_info:
                            parse_json('false')
                        assert exc_info.value.code == "INVALID_PAYLOAD"

                    def test_parse_json_null_instead_of_dict():
                        with pytest.raises(ApiError) as exc_info:
                            parse_json('null')
                        assert exc_info.value.code == "INVALID_PAYLOAD"

                    def test_parse_json_malformed_bracket():
                        with pytest.raises(ApiError) as exc_info:
                            parse_json('{"key": "value"')
                        assert exc_info.value.code == "INVALID_JSON"

                    def test_parse_json_trailing_comma():
                        with pytest.raises(ApiError) as exc_info:
                            parse_json('{"key": "value",}')
                        assert exc_info.value.code == "INVALID_JSON"

                    def test_parse_json_single_quotes():
                        with pytest.raises(ApiError) as exc_info:
                            parse_json("{'key': 'value'}")
                        assert exc_info.value.code == "INVALID_JSON"


