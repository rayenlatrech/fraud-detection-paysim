"""
Explore balance inconsistencies in PaySim dataset.
This helps detect data leakage.

Run from project root:

    python -m src.eda.explore_balances
"""

import pandas as pd

from src.config import PROCESSED_DATA_PATH


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Loaded data: {df.shape}")
    return df


def check_origin_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if oldbalanceOrg - amount == newbalanceOrg
    Returns a DataFrame of discrepancies.
    """
    expected_new_balance = df["oldbalanceOrg"] - df["amount"]
    discrepancy = expected_new_balance != df["newbalanceOrig"]

    df["origin_balance_ok"] = ~discrepancy

    return df


def check_dest_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if oldbalanceDest + amount == newbalanceDest
    """
    expected_new_balance = df["oldbalanceDest"] + df["amount"]
    discrepancy = expected_new_balance != df["newbalanceDest"]

    df["dest_balance_ok"] = ~discrepancy

    return df


def print_leakage_stats(df: pd.DataFrame) -> None:
    print("\n=== Origin balance rule broken ===")
    print(df["origin_balance_ok"].value_counts(normalize=True) * 100)

    print("\n=== Destination balance rule broken ===")
    print(df["dest_balance_ok"].value_counts(normalize=True) * 100)

    # Compare fraud vs non-fraud
    print("\n=== Origin balance rule broken by isFraud ===")
    print(df.groupby("isFraud")["origin_balance_ok"].value_counts(normalize=True))

    print("\n=== Destination balance rule broken by isFraud ===")
    print(df.groupby("isFraud")["dest_balance_ok"].value_counts(normalize=True))


def main():
    df = load_data()

    df = check_origin_balance(df)
    df = check_dest_balance(df)

    print_leakage_stats(df)


if __name__ == "__main__":
    main()
