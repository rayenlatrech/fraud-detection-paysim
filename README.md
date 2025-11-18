# PaySim Fraud Detection (6.3M Transactions)

This project implements a full **fraud detection pipeline** on the **PaySim mobile money transactions dataset** (~6.3 million rows).  
It covers data preprocessing, leakage removal, feature engineering, model training (Random Forest + XGBoost), model evaluation, SHAP explainability, and a full **Streamlit dashboard** with an interactive **what‑if fraud simulator**.

---

## 📊 Dataset
**PaySim1 dataset** (Kaggle): A synthetic mobile money transaction dataset containing:
- 6.3M transactions
- Transaction types: PAYMENT, CASH_IN, CASH_OUT, TRANSFER, DEBIT
- Features: amount, origin/destination balances, time (step), fraud label (`isFraud`)

➡️ Download: *Kaggle link provided in the project documentation.*

---

## 🧹 1. Data Processing & Leakage Removal

### Steps:
1. Loaded raw CSV (6.3M rows)
2. Removed **balance difference leakage**:
   - `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`  
   These columns directly expose whether a transfer was fraudulent (data leakage).
3. Converted:
   - categorical → numeric (one-hot)
   - created derived features:
     - `hour`, `day`, `log_amount`

### Scripts:
```
python -m src.data.make_dataset
python -m src.data.clean_dataset
```

---

## 🏗 2. Feature Engineering
From cleaned data, created:
- Time features: `hour`, `day`, `step`
- Transaction type one-hot:
  - `type_PAYMENT`
  - `type_CASH_IN`
  - `type_CASH_OUT`
  - `type_TRANSFER`
  - `type_DEBIT`
- Scaled features:
  - `log_amount`

### Script:
```
python -m src.features.build_features
```

---

## 🤖 3. Models (Random Forest & XGBoost)

### Random Forest
- Baseline strong non-linear model
- Handles imbalance using `class_weight='balanced'`

### XGBoost
- Tuned for imbalance using:
  ```
  scale_pos_weight = #negatives / #positives ≈ 974
  ```
- Tree method: **hist** (fast, scalable)
- Achieved:
  - **ROC-AUC ≈ 0.91**
  - **PR-AUC ≈ 0.33**

### Train scripts:
```
python -m src.models.train_model           # Random Forest
python -m src.models.train_xgboost_gpu    # XGBoost (CPU or GPU)
```

---

## 🧠 4. Explainability (SHAP)

### Global SHAP:
Shows which features matter most across all predictions.
Top features found:
- `type_PAYMENT` (strong non‑fraud signal)
- `type_CASH_IN`
- `type_CASH_OUT`
- `type_TRANSFER`
- `amount`, `log_amount`

### Local SHAP:
Per‑transaction explanations showing how each feature pushed the model toward:
- fraud (positive SHAP)
- or normal (negative SHAP)

### Scripts:
```
python -m src.explain.shap_xgboost
python -m src.explain.shap_single
```

---

## 📺 5. Streamlit Dashboard

Run the full interactive dashboard:
```
streamlit run app/streamlit/dashboard.py
```

### Dashboard features:
- **Global SHAP importance**
- **Transaction Explorer**
  - pick any real transaction from test set
  - view model prediction
  - view SHAP contributions
- **What‑If Fraud Simulator**
  Modify:
  - transaction type  
  - amount  
  - hour  
  - day  
  Dashboard recomputes:
  - prediction  
  - fraud probability  
  - SHAP explanation  

---

## 📁 Project Structure
```
fraud-detection-paysim/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   ├── clean_dataset.py
│   ├── features/
│   │   ├── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── train_xgboost_gpu.py
│   ├── explain/
│       ├── shap_xgboost.py
│       ├── shap_single.py
│
├── app/
│   ├── streamlit/
│       ├── dashboard.py
│
├── models/
│   ├── xgboost_gpu_model.json
│   ├── shap_global_importance.csv
│
└── README.md
```

---

## 🧪 Reproducing the Pipeline

### 1. Install environment
```
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
```

### 2. Run full pipeline
```
python -m src.data.make_dataset
python -m src.data.clean_dataset
python -m src.features.build_features
python -m src.models.train_xgboost_gpu
python -m src.explain.shap_xgboost
```

### 3. Launch dashboard
```
streamlit run app/streamlit/dashboard.py
```

---

## 📌 Skills Demonstrated
- Fraud detection & imbalanced learning  
- Data leakage detection  
- Large-scale data handling (6.3M rows)  
- Feature engineering  
- Random Forest & XGBoost modeling  
- ROC/PR analysis  
- SHAP explainability  
- Streamlit full interactive dashboard  
- What-if model simulation  
- Clean modular ML pipeline  

---

## 📜 License
MIT License.

---

## 👤 Author
**Rayen Latrech**  
Data Science Student  
GitHub: *add your GitHub link here*
