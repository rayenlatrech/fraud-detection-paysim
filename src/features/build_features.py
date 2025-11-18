"""
Feature engineering for PaySim clean dataset.

Run from project root:

    python -m src.features.build_features
"""

import pandas as pd
import numpy as np

from pathlib import Path

from src.config import DATA_DIR
from src.utils.io import ensure_dir

# input: cleaned dataset created earlier
CLEAN_DATA_PATH = DATA_DIR / "processed" / "paysim_cleaned.parquet"

# output: feature dataset
FEATURES_PATH = DATA_DIR / "processed" / "paysim_features.parquet"


def load_clean_data() -> pd.DataFrame:
    print(f"Loading cleaned data from: {CLEAN_DATA_PATH}")
    df = pd.read_parquet(CLEAN_DATA_PATH)
    print("Loaded:", df.shape)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    step goes from 1 to 744 (hours).
    We create:
    - hour of day (0–23)
    - day index (0–30)
    """
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    return df


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'type' categorical column.
    Drop the original 'type' column afterward.
    """
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)
    df = df.drop(columns=["type"])
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transformed amount features.
    """
    df["log_amount"] = np.log1p(df["amount"])
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put isFraud at the end for convenience.
    """
    target = df["isFraud"]
    df = df.drop(columns=["isFraud"])
    df["isFraud"] = target
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = encode_transaction_type(df)
    df = add_amount_features(df)

    # Drop raw ID columns and isFlaggedFraud (not usable as features)
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    print("\nDropping non-feature columns:", drop_cols)
    df = df.drop(columns=drop_cols)

    df = reorder_columns(df)
    return df


def main():
    ensure_dir(DATA_DIR / "processed")

    df = load_clean_data()
    df_features = build_features(df)

    print("\nFinal feature set:", df_features.shape)

    print(f"Saving feature dataset to: {FEATURES_PATH}")
    df_features.to_parquet(FEATURES_PATH, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
