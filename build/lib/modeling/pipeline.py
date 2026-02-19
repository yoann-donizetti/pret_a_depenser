from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Groupes de modèles
TREE_MODELS = (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

DENSE_MODELS = (
    HistGradientBoostingClassifier,
    MLPClassifier
)

SCALER_MODELS = (
    LogisticRegression,
    MLPClassifier
)

def build_pipeline(model, X):
    """
    Pipeline automatique adapté aux modèles utilisés dans ce projet :
    
    - LightGBM / XGBoost / CatBoost → pas de pipeline sklearn
    - RandomForest / ExtraTrees / GradientBoosting → OHE sparse, pas de scaler
    - HistGradientBoosting / MLP → OHE dense + scaler
    - LogisticRegression → scaler + OHE sparse
    """

    model_name = model.__class__.__name__

    # 1) Boosting natifs → pas de pipeline sklearn
    if model_name in ["LGBMClassifier", "XGBClassifier", "CatBoostClassifier"]:
        print(f" Boosting détecté ({model_name}) → pas de pipeline sklearn")
        return model

    # Colonnes
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    # 2) Arbres sklearn → OHE sparse, pas de scaler
    if isinstance(model, TREE_MODELS):
        print(f" Modèle arbre détecté ({model_name}) → OHE sparse, pas de scaler")
        num_transformer = "passthrough"
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # 3) Modèles denses (HGB, MLP) → OHE dense + scaler
    elif isinstance(model, DENSE_MODELS):
        print(f" Modèle dense détecté ({model_name}) → OHE dense + scaler")
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # 4) LogisticRegression → scaler + OHE sparse
    elif isinstance(model, SCALER_MODELS):
        print(f" Modèle nécessitant un scaler ({model_name}) → scaler + OHE sparse")
        num_transformer = StandardScaler(with_mean=False)
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # 5) Sécurité (ne devrait jamais arriver)
    else:
        print(f" Modèle non catégorisé ({model_name}) → scaler + OHE sparse")
        num_transformer = StandardScaler(with_mean=False)
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # Construction du préprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", cat_transformer, categorical_cols)
        ],
        sparse_threshold=1.0
    )

    # Pipeline finale
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ],
        memory="cache_sklearn"
    )

    return pipe