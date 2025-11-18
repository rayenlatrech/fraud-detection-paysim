"""
Clean PaySim dataset to remove known leakage columns.

Run:
    python -m src.data.clean_dataset
"""

import pandas as pd
from src.config import PROCESSED_DATA_PATH, DATA_DIR
from src.utils.io import write_parquet, ensure_dir


def load_data():
    print(f"Loading from {PROCESSED_DATA_PATH}")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print("Loaded:", df.shape)
    return df


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    leakage_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    print("\nDropping leakage columns:")
    print(leakage_columns)

    df = df.drop(columns=leakage_columns)
    print("After drop:", df.shape)

    return df


def main():
    df = load_data()
    df = remove_leakage(df)

    output_path = DATA_DIR / "processed" / "paysim_cleaned.parquet"
    print(f"Saving cleaned data to {output_path}")
    write_parquet(df, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
