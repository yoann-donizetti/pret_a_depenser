import json
from datetime import datetime
from pathlib import Path

import mlflow




def log_and_save(model_name, best_thr, df_thr, feature_set, kept_file, out_dir,COST_FN,COST_FP,FBETA_BETA):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv = out_dir / f"{model_name}_threshold_curve_{feature_set}_{ts}.csv"
    out_json = out_dir / f"{model_name}_best_threshold_{feature_set}_{ts}.json"

    # fichiers stables (sans timestamp)
    safe = model_name.lower()
    out_json_latest = out_dir / f"best_threshold_{safe}_{feature_set}.json"
    out_csv_latest  = out_dir / f"threshold_curve_{safe}_{feature_set}.csv"

    df_thr.to_csv(out_csv, index=False)
    df_thr.to_csv(out_csv_latest, index=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "feature_set": feature_set,
        "kept_file": str(kept_file),
        "business": {"cost_fn": COST_FN, "cost_fp": COST_FP, "fbeta_beta": FBETA_BETA},
        "threshold": float(best_thr["threshold"]),  # <-- clé simple
        "best": best_thr,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_json_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"{model_name}_threshold_valid_{feature_set}_{ts}"):
        mlflow.set_tag("phase", "threshold_optimization_valid")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("kept_file", str(kept_file))
        mlflow.set_tag("dataset", "valid_split")
        mlflow.set_tag("cost_fn", str(COST_FN))
        mlflow.set_tag("cost_fp", str(COST_FP))
        mlflow.set_tag("fbeta_beta", str(FBETA_BETA))

        # métriques float
        for k in ["threshold", "business_cost", "auc", "recall", "precision", "f1", f"fbeta_{FBETA_BETA}"]:
            if k in best_thr:
                mlflow.log_metric(f"best_{k}", float(best_thr[k]))

        # confusion int
        for k in ["tn", "fp", "fn", "tp"]:
            if k in best_thr:
                mlflow.log_metric(f"best_{k}", int(best_thr[k]))

        mlflow.log_artifact(str(out_csv))
        mlflow.log_artifact(str(out_json))
        mlflow.log_artifact(str(out_csv_latest))
        mlflow.log_artifact(str(out_json_latest))

    return out_csv_latest, out_json_latest, payload

