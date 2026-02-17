import numpy as np
import pandas as pd


def prepare_xgb(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare un DataFrame pour XGBoost :
    - sanitize des noms de colonnes (remplace [, ], <)
    - encodage des colonnes catégorielles en codes
    - conversion en float32
    """
    X = X.copy()

    # XGBoost interdit: [, ], <
    X.columns = [
        c.replace("[", "_").replace("]", "_").replace("<", "_")
        for c in X.columns
    ]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    X = X.astype(np.float32)

    return X


