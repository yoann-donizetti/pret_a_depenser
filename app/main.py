import time
from pathlib import Path
import json
import gradio as gr

from app.config import (
    MLFLOW_TRACKING_URI,
    ARTIFACT_ROOT,
    MODEL_URI,
    API_ARTIFACTS_URI,
)
from app.utils.io import load_txt_list, parse_json
from app.model.loader import setup_mlflow, download_run_artifacts, load_catboost_model
from app.model.predict import predict_score


# 1) Setup MLflow local
setup_mlflow(MLFLOW_TRACKING_URI, artifact_root=ARTIFACT_ROOT)

# 2) Télécharger les artefacts API du run (kept/cat/threshold)
cache_dir = ARTIFACT_ROOT / "_api_cache"
art_dir = download_run_artifacts(API_ARTIFACTS_URI, dst_dir=cache_dir)

# Noms attendus dans api_artifacts
kept_path = art_dir / "kept_features_top125_nocorr.txt"
cat_path  = art_dir / "cat_features_top125_nocorr.txt"
thr_path  = art_dir / "threshold_catboost_top125_nocorr.json"

KEPT_FEATURES = load_txt_list(kept_path)
CAT_FEATURES = load_txt_list(cat_path)
THRESHOLD = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])



# 3) Charger modèle (CatBoost natif)
MODEL = load_catboost_model(MODEL_URI)



def predict(json_text: str):
    t0 = time.time()
    try:
        payload = parse_json(json_text)
        out = predict_score(
            model=MODEL,
            payload=payload,
            kept_features=KEPT_FEATURES,
            cat_features=CAT_FEATURES,
            threshold=THRESHOLD,
        )
        out["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return out
    except Exception as e:
        return {"error": str(e)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Client features (JSON)", lines=14),
    outputs=gr.JSON(label="Prediction"),
    title="Prêt à Dépenser — Credit Scoring API",
    description="Colle un JSON (1 client) -> retourne proba_default (ex: 0.321) + décision.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)