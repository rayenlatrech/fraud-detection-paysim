"""
Basic EDA (Exploratory Data Analysis) for the PaySim dataset.

Run from project root (with venv activated):

    python -m src.eda.explore_data
"""

from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DATA_PATH


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed PaySim data saved by make_dataset.py
    """
    print(f"Loading processed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Loaded shape: {df.shape}")
    return df


def show_basic_info(df: pd.DataFrame) -> None:
    """
    Print basic dataframe info: columns, dtypes, etc.
    """
    print("\n=== DataFrame info ===")
    # .info() prints to stdout, so we don't wrap in print()
    df.info()

    print("\n=== First 5 rows ===")
    print(df.head())

    print("\n=== Numeric columns summary (describe) ===")
    print(df.describe())


def show_fraud_distribution(df: pd.DataFrame) -> None:
    """
    Show the distribution of the target 'isFraud' both in counts and percentages.
    This is essential to understand class imbalance.
    """
    print("\n=== 'isFraud' value counts (absolute) ===")
    counts = df["isFraud"].value_counts(dropna=False)
    print(counts)

    print("\n=== 'isFraud' value counts (percentage) ===")
    percentages = df["isFraud"].value_counts(normalize=True, dropna=False) * 100
    print(percentages.round(4))

    # Quick derived info: baseline accuracy if we always predict '0'
    if 0 in counts.index:
        total = len(df)
        majority_class_count = counts.loc[0]
        baseline_acc = majority_class_count / total * 100
        print(f"\nIf we always predict '0' (non-fraud), "
              f"baseline accuracy = {baseline_acc:.4f}% "
              f"(this is why accuracy is misleading here).")


def show_type_vs_fraud(df: pd.DataFrame) -> None:
    """
    For each transaction type, show how often it is fraud.
    This helps understand which types carry most fraud risk.
    """
    print("\n=== Fraud rate by transaction 'type' ===")
    # Crosstab between type and isFraud
    ctab = pd.crosstab(df["type"], df["isFraud"])

    print("\nRaw counts (rows = type, columns = isFraud):")
    print(ctab)

    print("\nRow-normalized (percentage of fraud per type):")
    ctab_norm = pd.crosstab(df["type"], df["isFraud"], normalize="index") * 100
    print(ctab_norm.round(4))


def main() -> None:
    df = load_processed_data()

    show_basic_info(df)
    show_fraud_distribution(df)
    show_type_vs_fraud(df)


if __name__ == "__main__":
    main()
