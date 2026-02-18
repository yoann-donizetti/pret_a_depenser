# app/main.py  (FastAPI + lifespan + create_app)

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import MODEL_URI, API_ARTIFACTS_URI, MLFLOW_TRACKING_URI
from app.model.loader import setup_mlflow, load_catboost_model, download_run_artifacts
from app.model.predict import predict_score
from app.utils.io import parse_json, load_txt_list
from app.utils.validation import validate_payload
from app.utils.errors import ApiError


# Globals (initialisés via lifespan)
MODEL = None
KEPT_FEATURES = None
CAT_FEATURES = None
THRESHOLD = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Init 1 seule fois au démarrage:
    - setup MLflow
    - load model
    - download artifacts (kept_features, cat_features, threshold)
    """
    global MODEL, KEPT_FEATURES, CAT_FEATURES, THRESHOLD

    setup_mlflow(MLFLOW_TRACKING_URI)
    MODEL = load_catboost_model(MODEL_URI)

    # dossier temporaire conservé pendant toute la durée du process
    tmpdir = tempfile.TemporaryDirectory()
    app.state._tmpdir = tmpdir  # garde une référence => pas supprimé
    cache_dir = Path(tmpdir.name)

    art_dir = download_run_artifacts(API_ARTIFACTS_URI, cache_dir)

    kept_path = art_dir / "kept_features_top125_nocorr.txt"
    cat_path = art_dir / "cat_features_top125_nocorr.txt"
    thr_path = art_dir / "threshold_catboost_top125_nocorr.json"

    KEPT_FEATURES = load_txt_list(kept_path)
    CAT_FEATURES = load_txt_list(cat_path)

    thr_obj = parse_json(thr_path.read_text(encoding="utf-8"))
    THRESHOLD = float(thr_obj["threshold"])

    # démarrage OK
    yield

    # shutdown: libérer le tempdir
    try:
        tmpdir.cleanup()
    except Exception:
        pass


def create_app(*, enable_lifespan: bool = True) -> FastAPI:
    """
    Factory d'app:
    - enable_lifespan=True  : prod (init modèle + artefacts)
    - enable_lifespan=False : tests (pas d'init lourde)
    """
    if enable_lifespan:
        app = FastAPI(
            title="Prêt à Dépenser — Credit Scoring API",
            description="POST /predict avec un JSON (1 client) -> proba défaut + décision + latence.",
            version="1.0.0",
            lifespan=lifespan,
        )
    else:
        app = FastAPI(
            title="Prêt à Dépenser — Credit Scoring API",
            description="POST /predict avec un JSON (1 client) -> proba défaut + décision + latence.",
            version="1.0.0",
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        ok = (
            MODEL is not None
            and KEPT_FEATURES is not None
            and CAT_FEATURES is not None
            and THRESHOLD is not None
        )
        return {"status": "ok" if ok else "not_ready"}

    @app.post("/predict")
    async def predict(payload: Dict[str, Any]) -> JSONResponse:
        t0 = time.time()
        try:
            if MODEL is None or KEPT_FEATURES is None or CAT_FEATURES is None or THRESHOLD is None:
                out = {"error": "NOT_READY","message": "API not ready: model/artifacts not loaded yet.","latency_ms": round((time.time() - t0) * 1000, 2),}
                return JSONResponse(status_code=503, content=out)

            payload_valid = validate_payload(
                payload,
                KEPT_FEATURES,
                CAT_FEATURES,
                reject_unknown_fields=True,
            )

            out = predict_score(
                MODEL,
                payload_valid,
                KEPT_FEATURES,
                CAT_FEATURES,
                threshold=THRESHOLD,
            )
            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            return JSONResponse(status_code=200, content=out)

        except ApiError as e:
            out = e.to_dict()
            out["latency_ms"] = round((time.time() - t0) * 1000, 2)
            return JSONResponse(status_code=e.http_status, content=out)

        except Exception as e:
            out = {
                "error": "INTERNAL_ERROR",
                "message": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }
            return JSONResponse(status_code=500, content=out)

    return app


# App utilisée par uvicorn: "app.main:app"
app = create_app(enable_lifespan=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app",host="127.0.0.1",port=int(os.getenv("PORT", "8000")),reload=True,)