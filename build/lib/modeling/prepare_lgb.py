import pandas as pd
def prepare_lgb_matrix(X: pd.DataFrame):
    """LightGBM: object/category -> category ; bool -> int8"""
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X2[c] = X2[c].astype("category")
    bool_cols = X2.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    for c in bool_cols:
        X2[c] = X2[c].astype("int8")
    return X2, cat_cols