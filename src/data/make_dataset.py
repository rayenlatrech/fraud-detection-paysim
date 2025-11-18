"""
Script to load the raw PaySim dataset, apply basic cleaning,
and save a first processed version for further feature engineering.

Run from project root:
    python -m src.data.make_dataset
"""

from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_DIR
from src.utils.io import ensure_dir, read_csv, write_parquet


def load_raw_data() -> pd.DataFrame:
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = read_csv(RAW_DATA_PATH)
    print(f"Loaded shape: {df.shape}")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df["type"] = df["type"].astype("category")
    df["nameOrig"] = df["nameOrig"].astype("category")
    df["nameDest"] = df["nameDest"].astype("category")
    return df


def main() -> None:
    ensure_dir(DATA_DIR)
    df = load_raw_data()

    print("Columns:", df.columns.tolist())
    print("Sample rows:")
    print(df.head())

    df_clean = basic_cleaning(df)

    print(f"Saving processed data to: {PROCESSED_DATA_PATH}")
    write_parquet(df_clean, PROCESSED_DATA_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
