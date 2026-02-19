import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from xgboost import XGBClassifier


from src.modeling.prepare_catboost import prepare_catboost

def cv_oof_proba_cb(params: dict, X: pd.DataFrame, y: pd.Series, n_splits=3, random_state=42) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X), dtype=float)

    X_m = prepare_catboost(X)
    cat_features = X_m.select_dtypes(include=["object"]).columns.tolist()  # CatBoost attend index/nom selon API, ici noms OK

    for tr_idx, va_idx in skf.split(X_m, y):
        X_tr, X_va = X_m.iloc[tr_idx], X_m.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)

        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return oof

def cv_oof_proba_lgb(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits=5,
    random_state=42,
    early_stopping_rounds=50,
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = LGBMClassifier(**params)

        # early stopping (accélère fortement certains trials)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )

        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return oof


def cv_oof_proba_xgb(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits=3,
    random_state=42,
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X), dtype=float)

    return oof
