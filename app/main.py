# app/main.py
from __future__ import annotations

import os
import time
import cProfile
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv

from app import config
from app.model.loader import load_bundle_from_hf, load_bundle_from_local
from app.model.predict import predict_score
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.utils.errors import ApiError
from app.utils.validation import validate_payload

from core.db.conn import init_db
from core.db.repo_features_store import get_features_by_id
from core.db.repo_prod_requests import insert_prod_request

load_dotenv()

MODEL = None
KEPT_FEATURES = None
CAT_FEATURES = None
THRESHOLD = None
CAT_COLS = None


def _bundle_source() -> str:
    src = (config.BUNDLE_SOURCE or "auto").strip().lower()
    if src in ("local", "hf"):
        return src
    return "hf" if config.HF_REPO_ID else "local"


def _safe_log(event: Dict[str, Any]) -> None:
    """Le logging ne doit jamais casser l'API."""
    try:
        insert_prod_request(event)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL, KEPT_FEATURES, CAT_FEATURES, CAT_COLS, THRESHOLD

    source = _bundle_source()

    if source == "hf":
        if not config.HF_REPO_ID:
            raise RuntimeError("HF_REPO_ID manquant (BUNDLE_SOURCE=hf)")
        MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD = load_bundle_from_hf(
            repo_id=config.HF_REPO_ID,
            model_path=config.HF_MODEL_PATH,
            kept_path=config.HF_KEPT_PATH,
            cat_path=config.HF_CAT_PATH,
            threshold_path=config.HF_THRESHOLD_PATH,
            token=config.HF_TOKEN,
        )
    else:
        MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD = load_bundle_from_local(
            model_path=config.LOCAL_MODEL_PATH,
            kept_path=config.LOCAL_KEPT_PATH,
            cat_path=config.LOCAL_CAT_PATH,
            threshold_path=config.LOCAL_THRESHOLD_PATH,
        )

    CAT_COLS = [c for c in (CAT_FEATURES or []) if c in (KEPT_FEATURES or [])]
    init_db()
    yield


def create_app(*, enable_lifespan: bool = True) -> FastAPI:
    app = FastAPI(
        title="Prêt à Dépenser — Credit Scoring API",
        version="1.0.0",
        lifespan=lifespan if enable_lifespan else None,
    )

    @app.get("/")
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ready = MODEL is not None and KEPT_FEATURES is not None and CAT_FEATURES is not None and THRESHOLD is not None
        return HealthResponse(status="ok" if ready else "not_ready")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest) -> JSONResponse:
        profiler = None
        if os.getenv("ENABLE_PROFILING", "0") == "1":
            profiler = cProfile.Profile()
            profiler.enable()

        t0 = time.time()
        sk_id = payload.SK_ID_CURR
        timing: Dict[str, float] = {}

        try:
            if MODEL is None:
                return JSONResponse(status_code=503, content={"error": "NOT_READY"})

            # DB
            features = get_features_by_id(int(sk_id))
            if not features:
                return JSONResponse(status_code=404, content={"error": "NOT_FOUND"})

            features["SK_ID_CURR"] = int(sk_id)

            # Validation (compatible tests: si kept_features vide/None, on skip)
            kept = KEPT_FEATURES or []
            cats = CAT_FEATURES or []
            if kept:
                payload_valid = validate_payload(
                    features,
                    kept,
                    cats,
                    reject_unknown_fields=True,
                )
            else:
                payload_valid = features

            # Threshold safe
            thr = float(THRESHOLD) if THRESHOLD is not None else 0.5

            out = predict_score(
                MODEL,
                payload_valid,
                kept_features=kept,
                cat_features=(CAT_COLS or []),
                threshold=thr,
                thread_count=1,
            )

            out["latency_ms"] = round((time.time() - t0) * 1000, 2)

            _safe_log(
                {
                    "endpoint": "/predict",
                    "status_code": 200,
                    "sk_id_curr": sk_id,
                    "latency_ms": out["latency_ms"],
                    "inputs": {"SK_ID_CURR": sk_id},
                    "outputs": {
                        "proba_default": out.get("proba_default"),
                        "score": out.get("score"),
                        "decision": out.get("decision"),
                        "threshold": out.get("threshold"),
                    },
                }
            )

            return JSONResponse(status_code=200, content=out)

        except ApiError as e:
            out = e.to_dict()
            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            _safe_log(
                {
                    "endpoint": "/predict",
                    "status_code": e.http_status,
                    "sk_id_curr": sk_id,
                    "latency_ms": out["latency_ms"],
                    "inputs": {"SK_ID_CURR": sk_id},
                    "error": out.get("error"),
                    "message": out.get("message"),
                    "outputs": {"details": out.get("details")},
                }
            )
            return JSONResponse(status_code=e.http_status, content=out)

        except Exception as e:
            out = {"error": "INTERNAL_ERROR", "message": str(e)}
            _safe_log(
                {
                    "endpoint": "/predict",
                    "status_code": 500,
                    "sk_id_curr": sk_id,
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                    "inputs": {"SK_ID_CURR": sk_id},
                    "error": out["error"],
                    "message": out["message"],
                }
            )
            print("PREDICT ERROR:", repr(e))
            return JSONResponse(status_code=500, content=out)

        finally:
            if profiler is not None:
                profiler.disable()

    return app


app = create_app(enable_lifespan=True)