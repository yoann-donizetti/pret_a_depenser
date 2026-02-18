from __future__ import annotations

import os
from pathlib import Path
import mlflow


def setup_mlflow(tracking_uri: str, artifact_root: Path | None = None) -> None:
    """
    Configure MLflow en local.
    artifact_root (optionnel) si tu veux forcer un dossier d'artefacts.
    """
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_ARTIFACT_URI"] = artifact_root.resolve().as_uri()


def download_run_artifacts(artifact_uri: str, dst_dir: Path) -> Path:
    """
    Télécharge un dossier d'artefacts MLflow (ex: runs:/.../api_artifacts)
    dans dst_dir et retourne le chemin local du dossier téléchargé.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=str(dst_dir))
    return Path(local_path)


def load_catboost_model(model_uri: str):
    """
    Charge le modèle en NATIF CatBoost via MLflow.
    Garantit predict_proba.
    """
    return mlflow.catboost.load_model(model_uri)