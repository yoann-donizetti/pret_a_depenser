# app/main.py
from __future__ import annotations

import os
import time
from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.model.loader import load_bundle_from_local, load_bundle_from_hf
from app.model.predict import predict_score
from app.utils.validation import validate_payload
from app.utils.errors import ApiError
from app.schemas import PredictRequest,HealthResponse


# Charge .env uniquement en dev/local (pas en prod Docker/CI)
if (os.getenv("ENV") or "dev").lower() != "prod":
    load_dotenv(override=False)


MODEL = None
KEPT_FEATURES = None
CAT_FEATURES = None
THRESHOLD = None


def _bundle_source() -> str:
    src = (os.getenv("BUNDLE_SOURCE") or "auto").strip().lower()
    if src in ("local", "hf"):
        return src
    # auto
    return "hf" if os.getenv("HF_REPO_ID") else "local"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD

    source = _bundle_source()

    if source == "hf":
        repo_id = os.getenv("HF_REPO_ID")
        if not repo_id:
            raise RuntimeError("HF_REPO_ID manquant (BUNDLE_SOURCE=hf)")

        token = os.getenv("HF_TOKEN")
        MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD = load_bundle_from_hf(
            repo_id=repo_id,
            model_path=os.getenv("HF_MODEL_PATH", "model/model.cb"),
            kept_path=os.getenv("HF_KEPT_PATH", "api_artifacts/kept_features_top125_nocorr.txt"),
            cat_path=os.getenv("HF_CAT_PATH", "api_artifacts/cat_features_top125_nocorr.txt"),
            threshold_path=os.getenv("HF_THRESHOLD_PATH", "api_artifacts/threshold_catboost_top125_nocorr.json"),
            token=token,
        )
    else:
        MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD = load_bundle_from_local(
            model_path=os.getenv("LOCAL_MODEL_PATH", "app/assets/model/model.cb"),
            kept_path=os.getenv("LOCAL_KEPT_PATH", "app/assets/api_artifacts/kept_features_top125_nocorr.txt"),
            cat_path=os.getenv("LOCAL_CAT_PATH", "app/assets/api_artifacts/cat_features_top125_nocorr.txt"),
            threshold_path=os.getenv("LOCAL_THRESHOLD_PATH", "app/assets/api_artifacts/threshold_catboost_top125_nocorr.json"),
        )

    yield


def create_app(*, enable_lifespan: bool = True) -> FastAPI:
    app = FastAPI(
        title="Prêt à Dépenser — Credit Scoring API",
        description="POST /predict avec un JSON (1 client) -> proba défaut + décision + latence.",
        version="1.0.0",
        lifespan=lifespan if enable_lifespan else None,
    )


    @app.get("/")
    def root():
        return {"message": "API running. Go to /docs"}
    

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ok = MODEL is not None and KEPT_FEATURES is not None and CAT_FEATURES is not None and THRESHOLD is not None
        return HealthResponse(status="ok" if ok else "not_ready")

    @app.post("/predict")
    async def predict(payload: PredictRequest) -> JSONResponse:
        t0 = time.time()
        try:
            payload_dict = payload.model_dump()

            if MODEL is None or KEPT_FEATURES is None or CAT_FEATURES is None or THRESHOLD is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "NOT_READY",
                        "message": "API not ready: model/artifacts not loaded yet.",
                        "latency_ms": round((time.time() - t0) * 1000, 2),
                    },
                )

            payload_valid = validate_payload(
                payload_dict, KEPT_FEATURES, CAT_FEATURES, reject_unknown_fields=True
            )

            out = predict_score(MODEL, payload_valid, KEPT_FEATURES, CAT_FEATURES, threshold=THRESHOLD)
            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            return JSONResponse(status_code=200, content=out)

        except ApiError as e:
            out = e.to_dict()
            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            return JSONResponse(status_code=e.http_status, content=out)

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": str(e),
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                },
            )

    return app  # <-- IMPORTANT: à ce niveau, pas dans un except/try

app = create_app(enable_lifespan=True)
