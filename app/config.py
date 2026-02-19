# app/config.py
from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key, default)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


# ENV
ENV = (_env("ENV", "dev") or "dev").lower()

# Source bundle: local | hf | auto
BUNDLE_SOURCE = (_env("BUNDLE_SOURCE", "auto") or "auto").lower()

# Local paths (assets gitignored)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(_env("ASSETS_DIR", str(PROJECT_ROOT / "app" / "assets")))

LOCAL_MODEL_PATH = _env("LOCAL_MODEL_PATH", str(ASSETS_DIR / "model" / "model.cb"))
LOCAL_KEPT_PATH = _env("LOCAL_KEPT_PATH", str(ASSETS_DIR / "api_artifacts" / "kept_features_top125_nocorr.txt"))
LOCAL_CAT_PATH = _env("LOCAL_CAT_PATH", str(ASSETS_DIR / "api_artifacts" / "cat_features_top125_nocorr.txt"))
LOCAL_THRESHOLD_PATH = _env("LOCAL_THRESHOLD_PATH", str(ASSETS_DIR / "api_artifacts" / "threshold_catboost_top125_nocorr.json"))

# Hugging Face
HF_REPO_ID = _env("HF_REPO_ID")
HF_TOKEN = _env("HF_TOKEN")

HF_MODEL_PATH = _env("HF_MODEL_PATH", "model/model.cb")
HF_KEPT_PATH = _env("HF_KEPT_PATH", "api_artifacts/kept_features_top125_nocorr.txt")
HF_CAT_PATH = _env("HF_CAT_PATH", "api_artifacts/cat_features_top125_nocorr.txt")
HF_THRESHOLD_PATH = _env("HF_THRESHOLD_PATH", "api_artifacts/threshold_catboost_top125_nocorr.json")