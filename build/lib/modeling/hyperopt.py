import optuna
import mlflow

from src.modeling.cv import cv_oof_proba_cb
from src.modeling.metrics import compute_metrics_and_cost



def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "rsm": trial.suggest_float("rsm", 0.7, 1.0),  # fraction de features
    }


import optuna
import mlflow

from src.modeling.metrics import compute_metrics_and_cost


def make_objective(
    X,
    y,
    base_params: dict,
    MODEL_NAME: str,
    FEATURE_SET_NAME: str,
    cv_oof_proba_fn,
    N_SPLITS: int = 3,
    RANDOM_STATE: int = 42,
    THRESH_FIXED: float = 0.5,
    COST_FN: int = 10,
    COST_FP: int = 1,
    FBETA_BETA: float = 3.0,
):
    """
    Crée et retourne la fonction objective(trial) compatible Optuna.
    """

    def objective(trial: optuna.Trial) -> float:

        params = dict(base_params)
        params.update(suggest_params(trial))

        oof = cv_oof_proba_fn(
            params=params,
            X=X,
            y=y,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )

        m = compute_metrics_and_cost(
            y_true=y,
            proba=oof,
            threshold=THRESH_FIXED,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
            beta=FBETA_BETA,
        )

        with mlflow.start_run(run_name=f"{MODEL_NAME}_trial_{trial.number:04d}"):

            mlflow.set_tag("phase", "hyperparam_optuna")
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("dataset", "train_split")
            mlflow.set_tag("feature_set", FEATURE_SET_NAME)
            mlflow.set_tag("n_splits", str(int(N_SPLITS)))
            mlflow.set_tag("threshold_fixed", str(float(THRESH_FIXED)))
            mlflow.set_tag("cost_fn", str(int(COST_FN)))
            mlflow.set_tag("cost_fp", str(int(COST_FP)))
            mlflow.set_tag("fbeta_beta", str(float(FBETA_BETA)))
            mlflow.set_tag("n_features", str(int(X.shape[1])))

            mlflow.log_params({f"cb.{k}": v for k, v in params.items()})

            mlflow.log_metric("train_cv.business_cost", m["business_cost"])
            mlflow.log_metric("train_cv.auc", m["auc"])
            mlflow.log_metric("train_cv.recall", m["recall"])
            mlflow.log_metric("train_cv.precision", m["precision"])
            mlflow.log_metric("train_cv.f1", m["f1"])
            mlflow.log_metric(f"train_cv.fbeta_{FBETA_BETA}", m[f"fbeta_{FBETA_BETA}"])
            mlflow.log_metric("train_cv.tn", m["tn"])
            mlflow.log_metric("train_cv.fp", m["fp"])
            mlflow.log_metric("train_cv.fn", m["fn"])
            mlflow.log_metric("train_cv.tp", m["tp"])

        return m["business_cost"]

    return objective