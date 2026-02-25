# app/main.py

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from app import config
from app.model.loader import load_bundle_from_hf, load_bundle_from_local
from app.model.predict import predict_score
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.utils.errors import ApiError
from app.utils.validation import validate_payload

from core.db.conn import init_db
from core.db.repo_features_store import get_features_by_id
from core.db.repo_prod_requests import insert_prod_request

from dotenv import load_dotenv

load_dotenv()

MODEL = None
KEPT_FEATURES = None
CAT_FEATURES = None
THRESHOLD = None


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
    global MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD

    # 1) charge modèle + artifacts
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

    # 2) init DB (idempotent)
    init_db()

    yield


def create_app(*, enable_lifespan: bool = True) -> FastAPI:
    app = FastAPI(
        title="Prêt à Dépenser — Credit Scoring API",
        description="POST /predict avec SK_ID_CURR -> proba défaut + décision + latence. Features lues depuis DB.",
        version="1.0.0",
        lifespan=lifespan if enable_lifespan else None,
    )

    @app.get("/")
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        t0 = time.time()
        ready = MODEL is not None and KEPT_FEATURES is not None and CAT_FEATURES is not None and THRESHOLD is not None
        status = "ok" if ready else "not_ready"
        latency_ms = round((time.time() - t0) * 1000, 2)

        _safe_log(
            {
                "endpoint": "/health",
                "status_code": 200,
                "latency_ms": latency_ms,
                "outputs": {"status": status},
            }
        )

        return HealthResponse(status=status)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest) -> JSONResponse:
        t0 = time.time()
        sk_id = payload.SK_ID_CURR

        # on stocke des timings "étape par étape" (sans changer la DB)
        timing: Dict[str, float] = {}

        # petit contexte utile pour le diagnostic
        bundle_source = _bundle_source()
        hf_repo_id = config.HF_REPO_ID if bundle_source == "hf" else None

        try:
            # not ready
            if MODEL is None or KEPT_FEATURES is None or CAT_FEATURES is None or THRESHOLD is None:
                out = {
                    "error": "NOT_READY",
                    "message": "API not ready: model/artifacts not loaded yet.",
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                }
                _safe_log(
                    {
                        "endpoint": "/predict",
                        "status_code": 503,
                        "sk_id_curr": sk_id,
                        "latency_ms": out["latency_ms"],
                        "inputs": {"SK_ID_CURR": sk_id},
                        "error": out["error"],
                        "message": out["message"],
                        "outputs": {
                            "timing": timing,
                            "bundle_source": bundle_source,
                            "hf_repo_id": hf_repo_id,
                        },
                    }
                )
                return JSONResponse(status_code=503, content=out)

            # 1) fetch features depuis DB
            t_db = time.time()
            features = get_features_by_id(int(sk_id))
            timing["db_ms"] = round((time.time() - t_db) * 1000, 2)

            if not features:
                out = {
                    "error": "NOT_FOUND",
                    "message": f"SK_ID_CURR={sk_id} introuvable dans features_store.",
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                }
                _safe_log(
                    {
                        "endpoint": "/predict",
                        "status_code": 404,
                        "sk_id_curr": sk_id,
                        "latency_ms": out["latency_ms"],
                        "inputs": {"SK_ID_CURR": sk_id},
                        "error": out["error"],
                        "message": out["message"],
                        "outputs": {
                            "timing": timing,
                            "bundle_source": bundle_source,
                            "hf_repo_id": hf_repo_id,
                        },
                    }
                )
                return JSONResponse(status_code=404, content=out)

            # 2) assure SK_ID_CURR présent dans les features (utile pour traçabilité interne)
            features["SK_ID_CURR"] = int(sk_id)

            # 3) validation "max"
            t_val = time.time()
            payload_valid = validate_payload(
                features,
                KEPT_FEATURES,
                CAT_FEATURES,
                reject_unknown_fields=True,
            )
            timing["validation_ms"] = round((time.time() - t_val) * 1000, 2)

            # 4) predict
            t_inf = time.time()
            out = predict_score(
                MODEL,
                payload_valid,
                KEPT_FEATURES,
                CAT_FEATURES,
                threshold=float(THRESHOLD),
            )
            timing["inference_ms"] = round((time.time() - t_inf) * 1000, 2)

            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            timing["total_ms"] = out["latency_ms"]

            _safe_log(
                {
                    "endpoint": "/predict",
                    "status_code": 200,
                    "sk_id_curr": sk_id,
                    "latency_ms": out["latency_ms"],
                    "inputs": payload_valid,
                    "outputs": {
                        "proba_default": out.get("proba_default"),
                        "score": out.get("score"),
                        "decision": out.get("decision"),
                        "threshold": out.get("threshold"),
                        "timing": timing,
                        "bundle_source": bundle_source,
                        "hf_repo_id": hf_repo_id,
                    },
                }
            )

            return JSONResponse(status_code=200, content=out)

        except ApiError as e:
            out = e.to_dict()
            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            timing["total_ms"] = out["latency_ms"]

            _safe_log(
                {
                    "endpoint": "/predict",
                    "status_code": e.http_status,
                    "sk_id_curr": sk_id,
                    "latency_ms": out["latency_ms"],
                    "inputs": {"SK_ID_CURR": sk_id},
                    "error": out.get("error"),
                    "message": out.get("message"),
                    "outputs": {
                        "details": out.get("details"),
                        "timing": timing,
                        "bundle_source": bundle_source,
                        "hf_repo_id": hf_repo_id,
                    },
                }
            )

            return JSONResponse(status_code=e.http_status, content=out)

        except Exception as e:
            out = {
                "error": "INTERNAL_ERROR",
                "message": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }
            timing["total_ms"] = out["latency_ms"]

            _safe_log(
                {
                    "endpoint": "/predict",
                    "status_code": 500,
                    "sk_id_curr": sk_id,
                    "latency_ms": out["latency_ms"],
                    "inputs": {"SK_ID_CURR": sk_id},
                    "error": out["error"],
                    "message": out["message"],
                    "outputs": {
                        "timing": timing,
                        "bundle_source": bundle_source,
                        "hf_repo_id": hf_repo_id,
                    },
                }
            )

            return JSONResponse(status_code=500, content=out)

    return app


app = create_app(enable_lifespan=True)