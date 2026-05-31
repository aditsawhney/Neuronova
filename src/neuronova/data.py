"""
data.py - Phase 1 data preparation

Loads the raw ADHD-200 phenotypic CSV, cleans features, binarises the target,
and saves a stratified 60/20/20 train/val/test split to data/processed/.

Run from the project root:
    uv run python src/neuronova/data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    DX contains numeric codes and the string 'pending'. Coerce to numeric
    (pending -> NaN), drop unknowns, then binarise: 0=control, 1=any ADHD subtype.
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
    Handedness is an EHI laterality quotient (-1 to +1), but some sites used
    categorical codes (0=Right, 1=Left, 2=Mixed) or 'L'. -999 = missing.

    Maps site-specific codes to EHI floats, then bins by standard thresholds:
        score < -0.4 -> Left, score > +0.4 -> Right, else -> Mixed
    """
    df = df.copy()
    s = df["Handedness"].astype(str).str.strip()

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
    """Coerce symptom score columns to float; -999 sentinel and non-numeric -> NaN."""
    df = df.copy()
    for col in ["Inattentive", "Hyper/Impulsive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-999, np.nan)
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
    """Stratified 60/20/20 split, locked at random_state=42."""
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