"""
Compute SHAP explanations for the trained XGBoost model.

Run from project root (venv active):

    python -m src.explain.shap_xgboost
"""

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier

from src.config import DATA_DIR, MODEL_DIR
from src.utils.io import ensure_dir

FEATURES_PATH = DATA_DIR / "processed" / "paysim_features.parquet"
MODEL_PATH = MODEL_DIR / "xgboost_gpu_model.json"


def load_features():
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
    print(f"Loading XGBoost model from: {MODEL_PATH}")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def compute_shap_global(model: XGBClassifier, X_background: pd.DataFrame, X_sample: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer and print global feature importance
    based on mean |SHAP|.
    """
    print("\nBuilding SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print(f"Computing SHAP values for {len(X_sample)} samples...")
    shap_values = explainer.shap_values(X_sample)

    # shap_values: (n_samples, n_features)
    # Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n=== Global SHAP feature importance (mean |SHAP|) ===")
    print(feature_importance)

    # Optionally save for later plotting / Streamlit
    out_path = MODEL_DIR / "shap_global_importance.csv"
    feature_importance.to_csv(out_path, index=False)
    print(f"\nSaved global SHAP importance to: {out_path}")


def main():
    df = load_features()
    X_train, X_test, y_train, y_test = time_based_split(df)

    # Use a smaller sample for SHAP to avoid huge computation
    # sample e.g. 5000 test rows
    n_background = min(2000, len(X_train))
    n_sample = min(5000, len(X_test))

    # sample without replacement
    background = X_train.sample(n=n_background, random_state=42)
    sample = X_test.sample(n=n_sample, random_state=42)

    model = load_model()

    compute_shap_global(model, background, sample)


if __name__ == "__main__":
    main()
