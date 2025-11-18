"""
Train an XGBoost model with GPU acceleration on PaySim features.

Run:
    python -m src.models.train_xgboost_gpu
"""

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.config import DATA_DIR, MODEL_DIR
from src.utils.io import ensure_dir

FEATURES_PATH = DATA_DIR / "processed" / "paysim_features.parquet"
MODEL_PATH = MODEL_DIR / "xgboost_gpu_model.json"


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    print("Loaded feature set:", df.shape)
    return df


def time_based_split(df, test_hours=24 * 7):
    max_step = df["step"].max()
    threshold = max_step - test_hours

    train_df = df[df["step"] <= threshold]
    test_df = df[df["step"] > threshold]

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"]

    return X_train, X_test, y_train, y_test


def train_xgboost_gpu(X_train, y_train):
    """
    Train XGBoost model.

    NOTE: Current xgboost build doesn't support GPU (gpu_hist not valid),
    so we fall back to 'hist' (fast CPU tree method).

    If you later install GPU-enabled XGBoost, you can change:
        tree_method='hist' -> 'gpu_hist'
        predictor='auto'   -> 'gpu_predictor'
    """

    print("\nTraining XGBoost model (CPU 'hist' tree method)...")

    # Compute scale_pos_weight based on imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight (neg/pos) = {scale_pos_weight:.2f}")

    model = XGBClassifier(
        tree_method="hist",        # <-- CPU fast method (change to 'gpu_hist' if GPU build installed)
        predictor="auto",          # <-- 'gpu_predictor' if GPU build
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        verbose=True,
    )

    return model



def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== ROC-AUC ===")
    print(roc_auc_score(y_test, y_scores))

    print("\n=== PR-AUC (Average Precision) ===")
    print(average_precision_score(y_test, y_scores))


def main():
    ensure_dir(MODEL_PATH.parent)

    df = load_features()
    X_train, X_test, y_train, y_test = time_based_split(df)

    model = train_xgboost_gpu(X_train, y_train)

    print("\nEvaluating...")
    evaluate(model, X_test, y_test)

    print(f"\nSaving model to: {MODEL_PATH}")
    model.save_model(MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
