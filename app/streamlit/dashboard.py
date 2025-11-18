"""
Streamlit dashboard for PaySim fraud model.

Run from project root (with venv active):

    streamlit run app/streamlit/dashboard.py
"""

from pathlib import Path
import sys

# ========= Make sure project root is on sys.path =========
# This lets us do `from src.config import ...` when running via Streamlit.
ROOT_DIR = Path(__file__).resolve().parents[2]  # D:\Projects\fraud-detection-paysim
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ================== Imports ==================
import numpy as np
import pandas as pd
import streamlit as st
import shap
from xgboost import XGBClassifier

from src.config import DATA_DIR, MODEL_DIR

FEATURES_PATH = DATA_DIR / "processed" / "paysim_features.parquet"
SHAP_GLOBAL_PATH = MODEL_DIR / "shap_global_importance.csv"
MODEL_PATH = MODEL_DIR / "xgboost_gpu_model.json"


# ================== Data & Model Loaders ==================

@st.cache_data
def load_shap_global() -> pd.DataFrame:
    """Load global SHAP feature importance (saved from shap_xgboost.py)."""
    return pd.read_csv(SHAP_GLOBAL_PATH)


@st.cache_data
def load_features() -> pd.DataFrame:
    """Load full feature dataset."""
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_model_and_explainer():
    """Load trained XGBoost model + SHAP TreeExplainer."""
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    return model, explainer


def time_based_test_split(df: pd.DataFrame, test_hours: int = 24 * 7):
    """
    Split into train/test using time-based split on 'step'.
    Last `test_hours` hours are used as test set.
    """
    max_step = df["step"].max()
    threshold = max_step - test_hours

    train_df = df[df["step"] <= threshold].copy()
    test_df = df[df["step"] > threshold].copy()

    return train_df, test_df


# ================== Streamlit UI ==================

def main():
    st.set_page_config(page_title="PaySim Fraud Model Dashboard", layout="wide")
    st.title("PaySim Fraud Detection – Explainability Dashboard")

    st.markdown(
        """
        This app visualizes an **XGBoost fraud detection model** trained on the PaySim dataset.
        It uses **SHAP values** to explain which features drive predictions globally and per transaction,
        and includes a **what-if simulator** to see how changes in inputs affect fraud risk.
        """
    )

    shap_df = load_shap_global()
    features_df = load_features()
    model, explainer = load_model_and_explainer()

    # IMPORTANT: feature columns (order) for model input
    feature_cols = [c for c in features_df.columns if c != "isFraud"]

    # ---------- Global SHAP Importance ----------
    st.header("🌍 Global Feature Importance (SHAP)")
    col_left, col_right = st.columns([2, 3])

    with col_left:
        top_n = st.slider(
            "Number of top features to display",
            min_value=3,
            max_value=len(shap_df),
            value=min(10, len(shap_df)),
            key="global_top_n",
        )
        top_features = shap_df.head(top_n)
        st.write("Mean absolute SHAP value per feature (higher = more global influence):")
        st.dataframe(top_features, use_container_width=True)

    with col_right:
        st.bar_chart(
            data=top_features.set_index("feature")["mean_abs_shap"]
        )

    # ---------- Dataset Info ----------
    st.subheader("🧾 Dataset Info")
    st.write(f"Total rows: **{len(features_df):,}**")
    st.write(f"Columns: `{list(features_df.columns)}`")

    train_df, test_df = time_based_test_split(features_df)
    num_fraud_test = int(test_df["isFraud"].sum())
    st.write(
        f"Test set (last 7 days): **{len(test_df):,}** rows "
        f"with **{num_fraud_test}** frauds."
    )

    st.markdown("---")

    # ---------- Transaction Explorer ----------
    st.header("🔍 Transaction Explorer (Local SHAP Explanation)")

    mode = st.radio(
        "Select how to choose a transaction:",
        ["By position in test set", "Random fraud sample", "Random non-fraud sample"],
        index=0,
        horizontal=True,
        key="tx_mode",
    )

    # Choose index based on mode
    if mode == "By position in test set":
        row_pos = st.slider(
            "Select a row index (position) in the test set",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            key="tx_row_pos",
        )
        x_row_full = test_df.iloc[row_pos]
        source_info = f"Test set position {row_pos}"
    else:
        if mode == "Random fraud sample":
            fraud_indices = test_df[test_df["isFraud"] == 1].index
            if len(fraud_indices) == 0:
                st.error("No fraud samples in test set.")
                return
            chosen_idx = int(np.random.choice(fraud_indices))
            source_info = f"Random fraud sample (index={chosen_idx})"
        else:  # Random non-fraud
            legit_indices = test_df[test_df["isFraud"] == 0].index
            if len(legit_indices) == 0:
                st.error("No non-fraud samples in test set.")
                return
            chosen_idx = int(np.random.choice(legit_indices))
            source_info = f"Random non-fraud sample (index={chosen_idx})"

        x_row_full = test_df.loc[chosen_idx]
        # For display, also get positional index within test_df
        row_pos = test_df.index.get_loc(chosen_idx)

    true_label = int(x_row_full["isFraud"])

    # Separate features from target
    X_row = x_row_full.drop(labels=["isFraud"]).to_frame().T
    X_row = X_row.astype(float)

    # Predict
    proba = model.predict_proba(X_row)[0, 1]
    pred_label = int(proba >= 0.5)

    st.subheader("🧠 Model Prediction for Selected Transaction")
    st.write(f"**Source:** {source_info}")
    st.write(f"**Row position in test set:** {row_pos}")
    st.write(f"**True label (`isFraud`):** `{true_label}`")
    st.write(f"**Predicted label (threshold 0.5):** `{pred_label}`")
    st.write(f"**Predicted fraud probability:** `{proba:.6f}`")

    with st.expander("Show raw feature values for this transaction"):
        st.dataframe(X_row.T, use_container_width=True)

    # Compute SHAP for this row
    shap_values = explainer.shap_values(X_row)[0]

    shap_table = pd.DataFrame({
        "feature": X_row.columns,
        "value": X_row.iloc[0].values,
        "shap_value": shap_values,
        "abs_shap": np.abs(shap_values),
    }).sort_values("abs_shap", ascending=False)

    st.subheader("📊 Feature Contributions (sorted by |SHAP|)")
    st.dataframe(shap_table, use_container_width=True)

    st.bar_chart(
        data=shap_table.set_index("feature")["shap_value"]
    )

    st.markdown(
        """
        **How to read this:**
        - Positive SHAP values push the prediction **toward fraud**.
        - Negative SHAP values push the prediction **toward non-fraud**.
        """
    )

    st.markdown("---")

    # ---------- What-if Simulator ----------
    st.header("🧪 What-if Simulator")

    st.markdown(
        """
        Use this panel to create a **synthetic transaction** by choosing type, amount, and time.
        The model will predict fraud probability and explain which features drove that prediction.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    # Use actual data stats for sliders
    min_amount = float(features_df["amount"].min())
    max_amount = float(features_df["amount"].quantile(0.99))  # avoid extreme 99.9+ outliers
    default_amount = float(features_df["amount"].median())

    with col1:
        tx_type = st.selectbox(
            "Transaction type",
            options=["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
            index=0,
        )

    with col2:
        amount = st.slider(
            "Amount",
            min_value=float(round(min_amount, 2)),
            max_value=float(round(max_amount, 2)),
            value=float(round(default_amount, 2)),
            step=100.0,
        )

    with col3:
        hour = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=12,
        )

    with col4:
        day = st.slider(
            "Day index (0–30)",
            min_value=0,
            max_value=30,
            value=15,
        )

    # Build a synthetic feature row consistent with training features
    # Columns: step, amount, hour, day, type_*, log_amount
    synthetic = {col: 0.0 for col in feature_cols}

    # step from day/hour (approx)
    synthetic["step"] = float(day * 24 + hour)
    synthetic["amount"] = float(amount)
    synthetic["hour"] = float(hour)
    synthetic["day"] = float(day)
    synthetic["log_amount"] = float(np.log1p(amount))

    # One-hot type columns
    type_cols = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
    for c in type_cols:
        synthetic[c] = 0.0

    if tx_type == "PAYMENT":
        synthetic["type_PAYMENT"] = 1.0
    elif tx_type == "CASH_OUT":
        synthetic["type_CASH_OUT"] = 1.0
    elif tx_type == "CASH_IN":
        synthetic["type_CASH_IN"] = 1.0
    elif tx_type == "TRANSFER":
        synthetic["type_TRANSFER"] = 1.0
    elif tx_type == "DEBIT":
        synthetic["type_DEBIT"] = 1.0

    # Create DataFrame in correct column order
    X_whatif = pd.DataFrame([synthetic])[feature_cols].astype(float)

    st.subheader("🔮 What-if Prediction")

    proba_whatif = model.predict_proba(X_whatif)[0, 1]
    pred_label_whatif = int(proba_whatif >= 0.5)

    st.write(f"**Chosen type:** `{tx_type}`")
    st.write(f"**Chosen amount:** `{amount:.2f}`")
    st.write(f"**Chosen hour/day:** `{hour}:00`, day `{day}`")
    st.write(f"**Predicted fraud label (0.5 threshold):** `{pred_label_whatif}`")
    st.write(f"**Predicted fraud probability:** `{proba_whatif:.6f}`")

    with st.expander("Show synthetic feature vector"):
        st.dataframe(X_whatif.T, use_container_width=True)

    # Compute SHAP for what-if row
    shap_whatif = explainer.shap_values(X_whatif)[0]

    shap_table_whatif = pd.DataFrame({
        "feature": X_whatif.columns,
        "value": X_whatif.iloc[0].values,
        "shap_value": shap_whatif,
        "abs_shap": np.abs(shap_whatif),
    }).sort_values("abs_shap", ascending=False)

    st.subheader("📈 What-if Feature Contributions")
    st.dataframe(shap_table_whatif, use_container_width=True)

    st.bar_chart(
        data=shap_table_whatif.set_index("feature")["shap_value"]
    )

    st.markdown(
        """
        Try changing:
        - **PAYMENT → TRANSFER/CASH_OUT** to see risk spike.
        - **Amount** from a few hundred to tens of thousands.
        - **Hour** (e.g. business hours vs night).
        """
    )


if __name__ == "__main__":
    main()
