from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key, default)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV = (_env("ENV", "dev") or "dev").lower()
BUNDLE_SOURCE = (_env("BUNDLE_SOURCE", "auto") or "auto").lower()

DATABASE_URL = _env("DATABASE_URL")

HF_REPO_ID = _env("HF_REPO_ID")
HF_TOKEN = _env("HF_TOKEN")