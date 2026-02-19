import time
import numpy as np
import mlflow

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

def train_with_cv(
    model,
    model_name,
    X,
    y,
    model_type="boosting",
    threshold=0.5,
    n_splits=5,
    random_state=42,
    log_fold_metrics=True,

    # --- business ---
    cost_fn=10,
    cost_fp=1,
    fbeta_beta=3,

    # --- robustesse / catégorielles ---
    fit_params=None,                 # dict passé à .fit()
    use_lgb_categorical=True,         # auto categorical_feature pour LGBM
    lgb_categorical_cols=None,        # optionnel: liste fournie sinon auto (dtype category)
):
    """
    Benchmark brut :
    - CV stratifiée
    - AUC (seuil-free)
    - Recall/Precision/F1/Fbeta + business cost au seuil fixe
    - MLflow: params + métriques + tags
    - Support LightGBM categorical_feature si X contient des colonnes category
    """

    if fit_params is None:
        fit_params = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    aucs, recalls, precisions, f1s, fbetas = [], [], [], [], []
    costs = []
    tns, fps, fns, tps = [], [], [], []

    print(f"\n===== Entraînement (benchmark CV) : {model_name} =====")
    start_time = time.time()

    with mlflow.start_run(run_name=model_name):
        # Tags
        mlflow.set_tag("phase", "benchmark_baseline")
        mlflow.set_tag("dataset", "train_split")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("cv_splits", int(n_splits))
        mlflow.set_tag("threshold_fixed", float(threshold))
        mlflow.set_tag("n_rows", int(X.shape[0]))
        mlflow.set_tag("n_features", int(X.shape[1]))

        mlflow.set_tag("cost_fn", int(cost_fn))
        mlflow.set_tag("cost_fp", int(cost_fp))
        mlflow.set_tag("fbeta_beta", float(fbeta_beta))

        # Params
        try:
            mlflow.log_params(model.get_params())
        except Exception:
            print("⚠️ Impossible de log les paramètres (get_params).")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n--- Fold {fold}/{n_splits} ---")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            m = clone(model)

            # Fit params fold-local
            local_fit_params = dict(fit_params)

            # LightGBM categorical_feature
            is_lgb = m.__class__.__name__.lower().startswith("lgbm")
            if use_lgb_categorical and is_lgb:
                if lgb_categorical_cols is not None:
                    cat_cols = list(lgb_categorical_cols)
                else:
                    cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()

                if len(cat_cols) > 0:
                    local_fit_params.setdefault("categorical_feature", cat_cols)
                    mlflow.set_tag("lgb_categorical_cols_n", int(len(cat_cols)))

            # Fit
            t0 = time.time()
            m.fit(X_train, y_train, **local_fit_params)
            fit_time = time.time() - t0

            # Probas
            t0 = time.time()
            proba = m.predict_proba(X_val)[:, 1]
            pred_time = time.time() - t0

            # Metrics
            auc = roc_auc_score(y_val, proba)

            y_pred = (proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            recall = recall_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            fbeta = fbeta_score(y_val, y_pred, beta=fbeta_beta, zero_division=0)

            business_cost = cost_fn * fn + cost_fp * fp

            aucs.append(auc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            fbetas.append(fbeta)
            costs.append(business_cost)

            tns.append(tn); fps.append(fp); fns.append(fn); tps.append(tp)

            print(
                f"   → AUC={auc:.4f} | "
                f"Recall@{threshold:.2f}={recall:.4f} | "
                f"F1@{threshold:.2f}={f1:.4f} | "
                f"F{fbeta_beta}@{threshold:.2f}={fbeta:.4f} | "
                f"Cost={business_cost}"
            )
            print(f"   → TN={tn} FP={fp} FN={fn} TP={tp} | fit={fit_time:.2f}s | pred={pred_time:.2f}s")

            if log_fold_metrics:
                mlflow.log_metric("fold_auc", auc, step=fold)
                mlflow.log_metric("fold_recall_fixed_threshold", recall, step=fold)
                mlflow.log_metric("fold_precision_fixed_threshold", precision, step=fold)
                mlflow.log_metric("fold_f1_fixed_threshold", f1, step=fold)
                mlflow.log_metric(f"fold_fbeta_{fbeta_beta}_fixed_threshold", fbeta, step=fold)

                mlflow.log_metric("fold_business_cost_fixed_threshold", business_cost, step=fold)
                mlflow.log_metric("fold_tn", tn, step=fold)
                mlflow.log_metric("fold_fp", fp, step=fold)
                mlflow.log_metric("fold_fn", fn, step=fold)
                mlflow.log_metric("fold_tp", tp, step=fold)

                mlflow.log_metric("fold_fit_time_sec", fit_time, step=fold)
                mlflow.log_metric("fold_pred_time_sec", pred_time, step=fold)

        # Aggregation
        auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
        recall_mean, recall_std = float(np.mean(recalls)), float(np.std(recalls))
        precision_mean, precision_std = float(np.mean(precisions)), float(np.std(precisions))
        f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
        fbeta_mean, fbeta_std = float(np.mean(fbetas)), float(np.std(fbetas))

        cost_mean, cost_std = float(np.mean(costs)), float(np.std(costs))

        tn_mean, fp_mean = float(np.mean(tns)), float(np.mean(fps))
        fn_mean, tp_mean = float(np.mean(fns)), float(np.mean(tps))

        total_time = float(time.time() - start_time)

        print("\n===== Résultats finaux (CV) =====")
        print(f"AUC                         : {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"Recall@{threshold:.2f}              : {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"Precision@{threshold:.2f}           : {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"F1@{threshold:.2f}                  : {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"F{fbeta_beta}@{threshold:.2f}                : {fbeta_mean:.4f} ± {fbeta_std:.4f}")
        print(f"Business cost (FN*{cost_fn}+FP*{cost_fp}) : {cost_mean:.2f} ± {cost_std:.2f}")
        print(f"TN/FP/FN/TP (moy)            : {tn_mean:.1f}/{fp_mean:.1f}/{fn_mean:.1f}/{tp_mean:.1f}")
        print(f"⏱ Temps total                : {total_time:.2f}s")

        # MLflow log
        mlflow.log_metric("auc_mean", auc_mean)
        mlflow.log_metric("auc_std", auc_std)

        mlflow.log_metric("recall_mean_fixed_threshold", recall_mean)
        mlflow.log_metric("recall_std_fixed_threshold", recall_std)

        mlflow.log_metric("precision_mean_fixed_threshold", precision_mean)
        mlflow.log_metric("precision_std_fixed_threshold", precision_std)

        mlflow.log_metric("f1_mean_fixed_threshold", f1_mean)
        mlflow.log_metric("f1_std_fixed_threshold", f1_std)

        mlflow.log_metric(f"fbeta_{fbeta_beta}_mean_fixed_threshold", fbeta_mean)
        mlflow.log_metric(f"fbeta_{fbeta_beta}_std_fixed_threshold", fbeta_std)

        mlflow.log_metric("business_cost_mean_fixed_threshold", cost_mean)
        mlflow.log_metric("business_cost_std_fixed_threshold", cost_std)

        mlflow.log_metric("tn_mean", tn_mean)
        mlflow.log_metric("fp_mean", fp_mean)
        mlflow.log_metric("fn_mean", fn_mean)
        mlflow.log_metric("tp_mean", tp_mean)

        mlflow.log_metric("train_time_sec", total_time)

    return {
        "model": model_name,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "recall_mean_fixed_threshold": recall_mean,
        "recall_std_fixed_threshold": recall_std,
        "precision_mean_fixed_threshold": precision_mean,
        "precision_std_fixed_threshold": precision_std,
        "f1_mean_fixed_threshold": f1_mean,
        "f1_std_fixed_threshold": f1_std,
        f"fbeta_{fbeta_beta}_mean_fixed_threshold": fbeta_mean,
        f"fbeta_{fbeta_beta}_std_fixed_threshold": fbeta_std,
        "business_cost_mean_fixed_threshold": cost_mean,
        "business_cost_std_fixed_threshold": cost_std,
        "threshold": float(threshold),
        "time_sec": total_time,
    }