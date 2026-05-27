"""
evaluate.py — Phase 3 evaluation and fairness audit

Loads the saved calibrated model and test set, computes metrics,
runs a bias audit by sex and age band, plots calibration and SHAP,
and saves a report to reports/.

Run from the project root:
    uv run --env-file .env python src/neuronova/evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# ── paths ──────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    meta = joblib.load(MODELS_DIR / "meta.joblib")
    test = pd.read_csv(PROCESSED / "test.csv")
    return model, meta, test


# ── core metrics ───────────────────────────────────────────────────────────

def evaluate_overall(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

    print("\n── Overall test metrics ─────────────────────────")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Avg precision:     {metrics['avg_precision']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Control", "ADHD"]))

    return metrics


# ── fairness audit ─────────────────────────────────────────────────────────

def audit_by_sex(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Gender in the dataset: 0.0 = male, 1.0 = female.
    This is the core equity check — girls are chronically underdiagnosed,
    so we want to confirm the model doesn't replicate that bias.
    """
    results = []
    gender_map = {0.0: "Male", 1.0: "Female"}

    for gender_val, label in gender_map.items():
        mask = X_test["Gender"] == gender_val
        if mask.sum() < 10:
            continue
        X_sub = X_test[mask]
        y_sub = y_test[mask]
        y_pred = model.predict(X_sub)
        y_prob = model.predict_proba(X_sub)[:, 1]

        results.append({
            "Group": label,
            "N": mask.sum(),
            "ADHD%": f"{y_sub.mean():.1%}",
            "F1": round(f1_score(y_sub, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_sub, y_prob) if y_sub.nunique() > 1 else float("nan"), 4),
        })

    df = pd.DataFrame(results)
    print("\n── Fairness audit — by sex ──────────────────────")
    print(df.to_string(index=False))
    return df


def audit_by_age(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Age bands chosen to reflect typical referral windows:
    - Under 10: early childhood, often missed
    - 10–13: primary school age, peak referral
    - 14+: adolescent, increasingly underdiagnosed
    """
    bins = [0, 10, 14, 100]
    labels = ["<10", "10–13", "14+"]
    age_band = pd.cut(X_test["Age"], bins=bins, labels=labels, right=False)

    results = []
    for band in labels:
        mask = age_band == band
        if mask.sum() < 10:
            continue
        X_sub = X_test[mask]
        y_sub = y_test[mask]
        y_pred = model.predict(X_sub)
        y_prob = model.predict_proba(X_sub)[:, 1]

        results.append({
            "Age band": band,
            "N": mask.sum(),
            "ADHD%": f"{y_sub.mean():.1%}",
            "F1": round(f1_score(y_sub, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_sub, y_prob) if y_sub.nunique() > 1 else float("nan"), 4),
        })

    df = pd.DataFrame(results)
    print("\n── Fairness audit — by age band ─────────────────")
    print(df.to_string(index=False))
    return df


# ── plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Control", "ADHD"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix — test set")
    plt.tight_layout()
    path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_roc_pr(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax1, name="VotingEnsemble")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_title("ROC curve — test set")

    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax2, name="VotingEnsemble")
    ax2.set_title("Precision-recall curve — test set")

    plt.tight_layout()
    path = REPORTS_DIR / "roc_pr.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_calibration(model, X_test, y_test):
    """
    Reliability diagram — predicted probability vs actual frequency.
    A perfectly calibrated model sits on the diagonal.
    This is why we wrapped the model in CalibratedClassifierCV.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, "s-", label="VotingEnsemble (calibrated)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve — test set")
    ax.legend()
    plt.tight_layout()
    path = REPORTS_DIR / "calibration.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Permutation importance — measures how much F1 drops when each feature
    is randomly shuffled. Works on any model including VotingClassifier.

    Repeating 10 times gives a distribution so we can show error bars,
    which is more honest than a single point estimate on 190 test samples.
    """
    feature_names = list(X_test.columns)

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    # Sort by mean importance descending
    order = result.importances_mean.argsort()[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_means = result.importances_mean[order]
    sorted_stds = result.importances_std[order]

    print("\n── Permutation importance (F1 drop) ────────────")
    for name, mean, std in zip(sorted_names, sorted_means, sorted_stds):
        print(f"  {name:20s}  {mean:.4f} ± {std:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        sorted_names[::-1], sorted_means[::-1],
        xerr=sorted_stds[::-1],
        align="center", color="steelblue", alpha=0.8, capsize=4,
    )
    ax.set_xlabel("Mean decrease in F1 when feature is shuffled")
    ax.set_title("Permutation feature importance — test set")
    plt.tight_layout()
    path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── report ─────────────────────────────────────────────────────────────────

def save_text_report(
    overall: dict,
    sex_audit: pd.DataFrame,
    age_audit: pd.DataFrame,
    model_name: str,
):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "audit_report.txt"
    with open(path, "w") as f:
        f.write("NeuroNova — Audit Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("Overall test metrics\n")
        f.write("-" * 30 + "\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFairness audit — by sex\n")
        f.write("-" * 30 + "\n")
        f.write(sex_audit.to_string(index=False))
        f.write("\n\nFairness audit — by age band\n")
        f.write("-" * 30 + "\n")
        f.write(age_audit.to_string(index=False))
        f.write("\n")
    print(f"\nSaved {path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading model and test set...")
    model, meta, test = load_artifacts()

    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    overall = evaluate_overall(model, X_test, y_test)
    sex_audit = audit_by_sex(model, X_test, y_test)
    age_audit = audit_by_age(model, X_test, y_test)

    print("\nGenerating plots...")
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_pr(model, X_test, y_test)
    plot_calibration(model, X_test, y_test)
    plot_permutation_importance(model, X_test, y_test)

    save_text_report(overall, sex_audit, age_audit, meta["model_name"])
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()