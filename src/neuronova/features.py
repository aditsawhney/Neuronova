"""
features.py - preprocessing pipeline

Defines the ColumnTransformer used by train.py. Never touches raw data directly.

Features:
    Numeric  : Gender (0/1), Age, Inattentive, Hyper/Impulsive
    Categoric: Handedness (Left / Right / Mixed)
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Gender", "Age", "Inattentive", "Hyper/Impulsive"]
CATEGORICAL_FEATURES = ["Handedness"]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns an unfitted ColumnTransformer.

    Numeric: median imputation + standard scaling.
    Categorical: most-frequent imputation + one-hot encoding (drop='first').
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # drop='first' avoids dummy variable trap; handle_unknown='ignore' for inference safety
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Returns readable feature names after fitting. Useful for SHAP and audit plots."""
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names