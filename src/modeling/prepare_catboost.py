import pandas as pd
def prepare_catboost(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("object").fillna("MISSING").astype(str)
    return X

import pandas as pd

def prepare_catboost_with_feature(X: pd.DataFrame):
    X_m = X.copy()

    cat_cols = X_m.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        X_m[col] = (
            X_m[col]
            .astype("object")
            .where(X_m[col].notna(), "__MISSING__")
            .astype(str)
        )

    # CatBoost accepte cat_features en noms de colonnes (comme tu fais)
    cat_features = cat_cols
    return X_m, cat_features

