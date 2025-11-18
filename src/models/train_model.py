"""
Baseline model training on PaySim features.

Run from project root (venv active):

    python -m src.models.train_model
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from src.config import DATA_DIR, MAIN_MODEL_PATH
from src.utils.io import ensure_dir

FEATURES_PATH = DATA_DIR / "processed" / "paysim_features.parquet"


def load_features() -> pd.DataFrame:
    print(f"Loading features from: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print("Loaded features shape:", df.shape)
    return df


def time_based_split(df: pd.DataFrame, test_hours: int = 24 * 7):
    """
    Use a time-based split instead of random splitting.
    Train on earlier steps, test on the last `test_hours` hours.
    """
    max_step = df["step"].max()
    threshold = max_step - test_hours

    train_df = df[df["step"] <= threshold].copy()
    test_df = df[df["step"] > threshold].copy()

    print(f"\nTime-based split with threshold step = {threshold}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"]

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a RandomForest baseline with class_weight to handle imbalance.
    We keep model relatively small so it can train on CPU in reasonable time.
    """
    print("\nTraining RandomForest baseline...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> None:
    """
    Evaluate with multiple metrics, focusing on the fraud class (1).
    """
    print("\nEvaluating model on test set...")

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function or just predictions
        y_scores = y_pred

    print("\n=== Classification report (fraud = class 1) ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    try:
        roc_auc = roc_auc_score(y_test, y_scores)
        print(f"\nROC-AUC: {roc_auc:.6f}")
    except ValueError:
        print("\nROC-AUC could not be computed (maybe only one class present).")

    try:
        pr_auc = average_precision_score(y_test, y_scores)
        print(f"PR-AUC (Average Precision): {pr_auc:.6f}")
    except ValueError:
        print("PR-AUC could not be computed.")


def main():
    ensure_dir(MAIN_MODEL_PATH.parent)

    df = load_features()
    X_train, X_test, y_train, y_test = time_based_split(df)

    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print(f"\nSaving model to: {MAIN_MODEL_PATH}")
    joblib.dump(model, MAIN_MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
