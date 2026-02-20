from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from app.utils.io import load_txt_list


def test_load_txt_list_basic():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        path.write_text("item1\nitem2\nitem3\n", encoding="utf-8")

        result = load_txt_list(path)
        assert result == ["item1", "item2", "item3"]


def test_load_txt_list_with_whitespace():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        path.write_text("  item1  \nitem2\n  item3\n", encoding="utf-8")

        result = load_txt_list(path)
        assert result == ["item1", "item2", "item3"]


def test_load_txt_list_with_empty_lines():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        path.write_text("item1\n\nitem2\n\n\nitem3\n", encoding="utf-8")

        result = load_txt_list(path)
        assert result == ["item1", "item2", "item3"]


def test_load_txt_list_empty_file():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        path.write_text("", encoding="utf-8")

        result = load_txt_list(path)
        assert result == []