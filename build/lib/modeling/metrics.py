from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
)
import numpy as np
import pandas as pd

def compute_metrics_and_cost(y_true, proba, threshold=0.5, cost_fn=10, cost_fp=1, beta=3) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = roc_auc_score(y_true, proba)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    business_cost = cost_fn * fn + cost_fp * fp

    return {
        "auc": float(auc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        f"fbeta_{beta}": float(fbeta),
        "business_cost": float(business_cost),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def compute_metrics(y_true, proba, threshold, cost_fn=10, cost_fp=1, beta=3):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    return {
        "threshold": float(threshold),
        "business_cost": float(cost_fn * fn + cost_fp * fp),
        "auc": float(roc_auc_score(y_true, proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"fbeta_{beta}": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def find_best_threshold(y_true, proba, thresholds, cost_fn=10, cost_fp=1, beta=3):
    rows = []
    for t in thresholds:
        rows.append(compute_metrics(y_true, proba, t, cost_fn=cost_fn, cost_fp=cost_fp, beta=beta))
    df = pd.DataFrame(rows)

    # tri: coût min, puis recall max, puis fbeta max
    df = df.sort_values(
        by=["business_cost", "recall", f"fbeta_{beta}", "auc"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return best, df