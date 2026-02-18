from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = (PROJECT_ROOT / "mlflow.db").resolve()
ARTIFACT_ROOT = (PROJECT_ROOT / "artifacts").resolve()
MLFLOW_TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"

# Run qui contient: model + api_artifacts (kept/cat/threshold)
RUN_ID = "88e88b90dfa34b44a9110a8035eae951"
MODEL_URI = f"runs:/{RUN_ID}/model"
API_ARTIFACTS_URI = f"runs:/{RUN_ID}/api_artifacts"