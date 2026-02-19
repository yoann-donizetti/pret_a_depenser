import pandas as pd
import numpy as np

def select_top_features(X: pd.DataFrame, fi: pd.DataFrame, top_n: int, importance_col: str) -> pd.DataFrame:
    """Select top-N features from fi, safely intersecting with X columns."""
    top_features = (
        fi.sort_values(importance_col, ascending=False)["feature"]
        .head(top_n)
        .tolist()
    )
    top_features = [c for c in top_features if c in X.columns]  # safe
    return X[top_features]


def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.9):
    """
    Drop highly correlated numerical/bool features using absolute Pearson corr.
    Returns: (X_final, to_drop, corr_matrix)
    """
    X = X.copy()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns
    X_num = X[num_cols].copy()

    if X_num.shape[1] < 2:
        return X, [], pd.DataFrame()

    corr = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if (upper[col] >= threshold).any()]
    X_final = X.drop(columns=to_drop, errors="ignore")
    return X_final, to_drop, corr