"""
train.py — Phase 2 model training

Builds a full sklearn Pipeline for each model (preprocessor + classifier),
runs GridSearchCV using a fixed validation fold (PredefinedSplit), selects
the best model by F1, calibrates it, and saves to models/.

Run from the project root:
    uv run python src/neuronova/train.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from neuronova.features import build_preprocessor

# ── paths ──────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")

RANDOM_STATE = 42


# ── model definitions ──────────────────────────────────────────────────────

def get_model_configs() -> list[dict]:
    """
    Each entry defines a model and its hyperparameter grid.
    Param keys use the pipeline step prefix: 'clf__<param>'.
    All models that support class_weight get 'balanced' to handle
    the 38/62 class imbalance without SMOTE overhead at this stage.
    """
    return [
        {
            "name": "LogisticRegression",
            "clf": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            "params": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__solver": ["lbfgs", "saga"],
            },
        },
        {
            "name": "LDA",
            "clf": LinearDiscriminantAnalysis(),
            "params": {
                "clf__solver": ["svd", "lsqr"],
                "clf__shrinkage": [None, "auto", 0.1, 0.5],
            },
        },
        {
            "name": "QDA",
            "clf": QuadraticDiscriminantAnalysis(),
            "params": {
                "clf__reg_param": [0.0, 0.1, 0.3, 0.5],
            },
        },
        {
            "name": "SVM",
            "clf": SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,  # needed for soft voting and calibration
                random_state=RANDOM_STATE,
            ),
            "params": {
                "clf__C": [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", "auto"],
            },
        },
        {
            "name": "RandomForest",
            "clf": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "params": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 5],
            },
        },
        {
            "name": "ExtraTrees",
            "clf": ExtraTreesClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "params": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 5],
            },
        },
        {
            "name": "NaiveBayes",
            "clf": GaussianNB(),
            "params": {
                "clf__var_smoothing": [1e-9, 1e-7, 1e-5],
            },
        },
    ]


# ── data loading ───────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    return train, val


def prepare_search_data(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Combines train and val into a single dataset for GridSearchCV.
    PredefinedSplit ensures the val rows are always the held-out fold —
    grid search never trains on val data, it only evaluates on it.

    Returns X_combined, y_combined, and the split indices array.
    """
    combined = pd.concat([train, val], ignore_index=True)
    X = combined.drop(columns=["target"])
    y = combined["target"]

    # -1 = training fold, 0 = validation fold
    split_indices = np.array([-1] * len(train) + [0] * len(val))

    return X, y, split_indices


# ── training ───────────────────────────────────────────────────────────────

def train_all_models(
    X: pd.DataFrame, y: pd.Series, split_indices: np.ndarray
) -> list[dict]:
    """
    Trains every model config via GridSearchCV with a fixed val fold.
    All models scored on F1 (macro) — consistent across the board.
    Returns results sorted by val F1 descending.
    """
    cv = PredefinedSplit(test_fold=split_indices)
    results = []

    for config in get_model_configs():
        name = config["name"]
        print(f"Training {name}...", end=" ", flush=True)

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", config["clf"]),
        ])

        # LDA with svd solver doesn't support shrinkage — skip invalid combos
        search = GridSearchCV(
            pipeline,
            param_grid=config["params"],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            error_score=0.0,  # invalid param combos score 0 rather than crashing
        )
        search.fit(X, y)

        best_f1 = search.best_score_
        print(f"best val F1 = {best_f1:.4f} | params = {search.best_params_}")

        results.append({
            "name": name,
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "val_f1": best_f1,
        })

    results.sort(key=lambda r: r["val_f1"], reverse=True)
    return results


def build_and_save_ensemble(
    results: list[dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 3,
) -> tuple[VotingClassifier, float]:
    """
    Calibrate each top pipeline individually, then combine into a soft
    voting ensemble.

    Calibrating pipelines individually (rather than wrapping VotingClassifier
    in CalibratedClassifierCV) keeps the preprocessor attached to each
    sub-estimator, so string inputs like Handedness reach the encoder intact.
    """
    from sklearn.metrics import f1_score

    top = results[:top_n]
    print(f"\nCalibrating and building ensemble from: {[r['name'] for r in top]}")

    calibrated_estimators = []
    for r in top:
        print(f"  Calibrating {r['name']}...", end=" ", flush=True)
        cal = CalibratedClassifierCV(r["best_estimator"], method="sigmoid", cv=5)
        cal.fit(X_train, y_train)
        calibrated_estimators.append((r["name"], cal))
        print("done")

    ensemble = VotingClassifier(estimators=calibrated_estimators, voting="soft")
    ensemble.fit(X_train, y_train)

    ensemble_f1 = f1_score(y_val, ensemble.predict(X_val))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(ensemble, MODELS_DIR / "model.joblib")
    print(f"Saved ensemble to models/model.joblib  (val F1 = {ensemble_f1:.4f})")

    meta = {
        "model_name": "VotingEnsemble",
        "features": ["Gender", "Age", "Handedness", "Inattentive", "Hyper/Impulsive"],
    }
    joblib.dump(meta, MODELS_DIR / "meta.joblib")

    return ensemble, ensemble_f1


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading splits...")
    train, val = load_splits()
    X, y, split_indices = prepare_search_data(train, val)
    print(f"Combined search set: {len(X)} rows ({len(train)} train + {len(val)} val)\n")

    results = train_all_models(X, y, split_indices)

    print("\n── Results summary ──────────────────────────────")
    for r in results:
        print(f"  {r['name']:20s}  val F1 = {r['val_f1']:.4f}")

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_val = val.drop(columns=["target"])
    y_val = val["target"]

    ensemble, ensemble_f1 = build_and_save_ensemble(
        results, X_train, y_train, X_val, y_val, top_n=3
    )
    print(f"  {'VotingEnsemble':20s}  val F1 = {ensemble_f1:.4f}")
    print(f"\nWinner: VotingEnsemble (val F1 = {ensemble_f1:.4f})")


if __name__ == "__main__":
    main()