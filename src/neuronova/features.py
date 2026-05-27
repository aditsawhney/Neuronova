"""
features.py — preprocessing pipeline definition

Defines the ColumnTransformer that handles imputation, encoding, and scaling
for all features. This is imported by train.py and composed with each model
into a full sklearn Pipeline — nothing here ever touches raw data directly.

Feature inventory (post-audit):
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

    Numeric pipeline:
        1. Median imputation — robust to the skew in behaviour scores,
           and correct for the ~281 missing Inattentive/Hyper/Impulsive values.
        2. Standard scaling — required for LR, LDA, SVM. Harmless for trees.

    Categorical pipeline:
        1. Most-frequent imputation — only 9 missing Handedness values.
        2. One-hot encoding with drop='first' to avoid dummy variable trap.
           Left/Mixed/Right → 2 binary columns.
           handle_unknown='ignore' so unseen categories at inference don't crash.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # silently drops anything not listed — safety net
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Returns human-readable feature names after transformation.
    Useful for SHAP plots and the audit report.
    Call this after the preprocessor has been fitted.
    """
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names