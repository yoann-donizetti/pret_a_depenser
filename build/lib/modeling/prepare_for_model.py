import pandas as pd

def prepare_application_for_model(df, model_type="boosting", target_col="TARGET"):
    df = df.copy()

    # Séparation X / y
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # 1) Colonnes 100% NA
    nan_cols = X.columns[X.isna().mean() == 1.0]
    if len(nan_cols) > 0:
        X = X.drop(columns=nan_cols)

    # 2) Colonnes constantes
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)

    # 3) Bool -> int8 (pratique pour sklearn et boosting)
    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype("int8")

    # --- BOOSTING ---
    if model_type == "boosting":
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            X[cat_cols] = X[cat_cols].astype("category")
        return X, y

    # --- SKLEARN ---
    if model_type == "sklearn":
        # IMPORTANT : on ne remplit plus les NA ici
        # et on ne rajoute plus APP_HAS_NA (add_indicator le fait mieux)
        return X, y

    raise ValueError("model_type doit être 'boosting' ou 'sklearn'")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def make_preprocessor(X: pd.DataFrame):
    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32", "int8"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in bool_cols]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler(with_mean=False)),  # OK même si sortie sparse
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),  # sparse_output=True par défaut
    ])

    bool_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # optionnel: ("to_int", FunctionTransformer(lambda a: a.astype("int8")))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
            ("bool", bool_pipe, bool_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, {"num": num_cols, "cat": cat_cols, "bool": bool_cols}
