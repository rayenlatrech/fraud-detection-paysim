# Fraud Detection on 6.3M Mobile-Money Transactions (PaySim)

An end-to-end fraud detection pipeline on the PaySim dataset: data cleaning,
feature engineering, a time-based evaluation, Random Forest and XGBoost models for a
heavily imbalanced target (about 0.13% fraud), SHAP explainability, and a Streamlit
dashboard with a what-if simulator.

## Results (XGBoost, last 7 days held out)

| Metric | Value |
|---|---|
| ROC-AUC | 0.91 |
| PR-AUC (average precision) | 0.33 |

With roughly one fraud per 1,000 transactions, accuracy and even ROC-AUC look
flattering, so **PR-AUC is the metric that matters here**. For reference, a random
classifier's PR-AUC equals the fraud rate in the test window.

These numbers are deliberately obtained **without** the account-balance columns (see
below). With those columns, models on this dataset commonly reach near-perfect scores,
which says more about the simulator than about fraud detection.

## Key design decisions

**1. Removing the balance columns.**
PaySim includes `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest` and
`newbalanceDest`. In this simulator, fraudulent transfers produce very distinctive
balance patterns (for example, accounts emptied to exactly zero), and models trained on
them reach near-perfect scores. That reflects how the data was generated rather than
a signal a real bank could rely on, so I dropped them to make the task realistic and
to see what the model can learn from the transaction itself.

**2. Time-based split instead of a random split.**
The last 7 days (168 hourly steps) form the test set. A random split would let the
model train on transactions from the same hours it is tested on, which overstates
performance in deployment.

**3. Handling a 1:1000 class imbalance.**
Random Forest uses `class_weight="balanced"`; XGBoost uses
`scale_pos_weight = negatives / positives ≈ 974`.

## Features

| Feature | Description |
|---|---|
| `type_*` | One-hot transaction type (PAYMENT, CASH_IN, CASH_OUT, TRANSFER, DEBIT) |
| `amount`, `log_amount` | Transaction amount, raw and log-scaled |
| `hour` | Hour of day (`step % 24`) |
| `day`, `step` | Day index and hourly step of the simulation |

## Explainability (SHAP)

Global mean |SHAP| ranks the transaction type first: in PaySim, fraud only occurs
in TRANSFER and CASH_OUT, so `type_PAYMENT` and `type_CASH_IN` are strong
"not fraud" signals. Amount and hour of day follow. Full ranking:
[`models/shap_global_importance.csv`](models/shap_global_importance.csv).

Local SHAP explains individual predictions, showing which features pushed a given
transaction toward or away from fraud.

## Dashboard

```bash
streamlit run app/streamlit/dashboard.py
```

- **Global importance**: SHAP ranking of all features.
- **Transaction explorer**: pick a real test transaction and see its fraud probability and SHAP breakdown.
- **What-if simulator**: change the type, amount, hour and day of a transaction and watch the prediction and explanation update.

## Known limitations

- `step` and `day` are used as features while the split is also based on time, so
  every test transaction has a `step` and `day` value the model never saw in training.
  Dropping them would make the evaluation cleaner.
- PaySim is synthetic. Transaction volume and fraud are not distributed over time the
  way they would be in real data, so the time features partly learn simulator artifacts.
- No per-account history features (transaction velocity, typical amount per
  sender), which is where most real-world fraud signal comes from.

## Project structure

```
├── src/
│   ├── config.py                 # paths
│   ├── data/                     # load raw CSV, drop balance columns
│   ├── eda/                      # exploratory analysis
│   ├── features/                 # feature engineering
│   ├── models/                   # Random Forest, XGBoost training
│   └── explain/                  # global and local SHAP
├── app/streamlit/dashboard.py    # interactive dashboard
└── models/                       # trained models, SHAP importance
```

## How to run

1. Download the PaySim dataset from Kaggle
   ([`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1)) and place the
   CSV at `data/raw/PS_20174392719_1491204439457_log.csv`.
2. Create an environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```

3. Run the pipeline from the project root:

   ```bash
   python -m src.data.make_dataset        # raw CSV -> parquet
   python -m src.data.clean_dataset       # drop balance columns
   python -m src.features.build_features  # feature engineering
   python -m src.models.train_model       # Random Forest baseline (optional)
   python -m src.models.train_xgboost_gpu # XGBoost (runs on CPU by default)
   python -m src.explain.shap_xgboost     # global SHAP importance
   streamlit run app/streamlit/dashboard.py
   ```

## License

MIT

## Author

Rayen Latrech · [GitHub](https://github.com/rayenlatrech)
