"""
Local SHAP explanations for individual PaySim transactions.

Run from project root (venv active):

    python -m src.explain.shap_single

This will:
- load the trained XGBoost model
- load the feature data
- do the same time-based split
- pick one fraud and one non-fraud transaction
- compute and print SHAP values for each, sorted by |SHAP|
"""

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from src.config import DATA_DIR, MODEL_DIR

FEATURES_PATH = DATA_DIR / "processed" / "paysim_features.parquet"
MODEL_PATH = MODEL_DIR / "xgboost_gpu_model.json"


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    print("Loaded features:", df.shape)
    return df


def time_based_split(df: pd.DataFrame, test_hours: int = 24 * 7):
    max_step = df["step"].max()
    threshold = max_step - test_hours

    train_df = df[df["step"] <= threshold].copy()
    test_df = df[df["step"] > threshold].copy()

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"]

    return X_train, X_test, y_train, y_test


def load_model() -> XGBClassifier:
    print(f"Loading model from: {MODEL_PATH}")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def explain_single(model: XGBClassifier, x_row: pd.Series, label: int, index_name: str):
    """
    Compute SHAP values for a single row and print contributions.
    """
    print(f"\n=== SHAP explanation for {index_name} (true label = {label}) ===")

    # XGBoost expects 2D array
    # XGBoost expects 2D array with numeric dtypes
    X = x_row.to_frame().T

    # Ensure all columns are numeric (float)
    X = X.astype(float)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]  # 1D array (n_features,)


    # Build a table: feature, value, shap, |shap|
    result = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[0].values,
        "shap_value": shap_values,
        "abs_shap": np.abs(shap_values),
    }).sort_values("abs_shap", ascending=False)

    print(result)

    return result


def main():
    df = load_features()
    X_train, X_test, y_train, y_test = time_based_split(df)

    model = load_model()

    # pick one fraud example and one non-fraud example from test set
    fraud_idx = y_test[y_test == 1].index[0]
    legit_idx = y_test[y_test == 0].index[0]

    x_fraud = X_test.loc[fraud_idx]
    x_legit = X_test.loc[legit_idx]

    explain_single(model, x_fraud, label=1, index_name=f"fraud sample (idx={fraud_idx})")
    explain_single(model, x_legit, label=0, index_name=f"non-fraud sample (idx={legit_idx})")


if __name__ == "__main__":
    main()
