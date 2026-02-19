import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
from pathlib import Path
import numpy as np

def log_and_register_baseline(
    model_name: str,
    registry_name: str,
    model_obj,
    params: dict,
    metrics: dict,
    threshold: float,
    feature_set: str,
    kept_file: Path,
    COST_FN,
    COST_FP,
    FBETA_BETA
):
    with mlflow.start_run(run_name=f"BASELINE_{model_name}"):

        # tags
        mlflow.set_tag("phase", "baseline_registry")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("kept_file", str(kept_file))
        mlflow.set_tag("dataset_train", "train+valid")
        mlflow.set_tag("dataset_test", "test_split")

        mlflow.set_tag("threshold_business", str(float(threshold)))
        mlflow.log_metric("threshold_business", float(threshold))

        mlflow.set_tag("cost_fn", str(COST_FN))
        mlflow.set_tag("cost_fp", str(COST_FP))
        mlflow.set_tag("fbeta_beta", str(FBETA_BETA))

        # params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # metrics test
        for k, v in metrics.items():
            if k == "threshold":
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                mlflow.log_metric(f"test.{k}", float(v))
            else:
                mlflow.set_tag(f"test.{k}", str(v))

        # register model
        if model_name == "LightGBM":
            mlflow.lightgbm.log_model(
                lgb_model=model_obj,
                artifact_path="model",
                registered_model_name=registry_name
            )
        elif model_name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=model_obj,
                artifact_path="model",
                registered_model_name=registry_name
            )
        elif model_name == "CatBoost":
            mlflow.catboost.log_model(
                cb_model=model_obj,
                artifact_path="model",
                registered_model_name=registry_name
            )
        else:
            raise ValueError("model_name inconnu")