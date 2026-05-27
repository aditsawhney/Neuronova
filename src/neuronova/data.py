"""
data.py — Phase 1 data foundation

Loads the raw ADHD-200 phenotypic CSV, applies the feature audit,
binarises the target, and produces a single locked stratified split.
Processed sets are saved to data/processed/ as CSVs.

Run from the project root:
    uv run python src/neuronova/data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────
RAW = Path("data/raw/adhd200_preprocessed_phenotypics.csv")
PROCESSED = Path("data/processed")

KEEP_FEATURES = ["Gender", "Age", "Handedness", "Inattentive", "Hyper/Impulsive"]
TARGET_COL = "DX"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    DX is a mixed-type column — numeric codes plus the string 'pending'.
    Coerce to numeric first (pending → NaN), then drop unknowns.

    Coding: 0 = control, 1 = combined, 2 = inattentive, 3 = hyperactive/impulsive.
    We binarise: 0 → 0, any ADHD subtype → 1.
    """
    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    n_pending = df[TARGET_COL].isna().sum()
    print(f"\nDropping {n_pending} rows with unparseable DX (e.g. 'pending')")
    df = df.dropna(subset=[TARGET_COL])

    print("DX value counts (after coercion):")
    print(df[TARGET_COL].value_counts().sort_index())

    df["target"] = (df[TARGET_COL] != 0).astype(int)
    print(f"\nTarget distribution after binarisation:")
    print(df["target"].value_counts())
    print(f"Class balance: {df['target'].mean():.1%} ADHD")
    return df


def clean_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handedness in ADHD-200 is an Edinburgh Handedness Inventory (EHI) laterality
    quotient — a float from -1.0 (strongly left) to +1.0 (strongly right).
    Some sites used a different coding: 0=right, 1=left, 2=mixed, L=left.
    -999 is a sentinel for missing.

    Strategy:
    - Map site-specific categorical codes to EHI-equivalent floats first.
    - Coerce everything to float (-999 and unparseable → NaN).
    - Bin into Left / Right / Mixed using standard EHI thresholds:
        score < -0.4  → Left
        score > +0.4  → Right
        otherwise     → Mixed
    """
    df = df.copy()
    s = df["Handedness"].astype(str).str.strip()

    # Dataset key: 0=Left, 1=Right, 2=Ambidextrous. and 'L' treated as Left.
    site_map = {"0": "-1.0", "1": "1.0", "2": "0.0", "3": "0.0", "L": "-1.0"}
    s = s.replace(site_map)

    df["Handedness"] = pd.to_numeric(s, errors="coerce")
    df.loc[df["Handedness"] == -999, "Handedness"] = np.nan

    def ehi_to_label(x):
        if pd.isna(x):
            return np.nan
        if x < -0.4:
            return "Left"
        if x > 0.4:
            return "Right"
        return "Mixed"

    df["Handedness"] = df["Handedness"].map(ehi_to_label)
    print(f"\nHandedness distribution after cleaning:")
    print(df["Handedness"].value_counts(dropna=False))
    return df


def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inattentive and Hyper/Impulsive are stored as strings in some sites.
    Coerce to float — any non-numeric value becomes NaN (handled by imputer
    in the pipeline later).
    """
    df = df.copy()
    for col in ["Inattentive", "Hyper/Impulsive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-999, np.nan)  # sentinel → NaN
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = KEEP_FEATURES + ["target"]
    return df[cols].copy()


def inspect_features(df: pd.DataFrame) -> None:
    print("\nFeature overview:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nGender values: {df['Gender'].unique()}")


def split_and_save(df: pd.DataFrame) -> None:
    """
    Single stratified split, locked.
    60% train / 20% val / 20% test
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out = X_split.copy()
        out["target"] = y_split.values
        out.to_csv(PROCESSED / f"{split_name}.csv", index=False)
        print(f"{split_name:5s}: {len(out):4d} rows | {y_split.mean():.1%} ADHD")

    print(f"\nProcessed sets saved to {PROCESSED}/")


def main():
    df = load_raw()
    df = clean_target(df)
    df = clean_handedness(df)
    df = clean_numeric_features(df)
    df = select_features(df)
    inspect_features(df)
    split_and_save(df)


if __name__ == "__main__":
    main()