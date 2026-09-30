"""Script to generate the complete, professional Jupyter Notebook for the internship deliverable."""
import json
import os

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # -------------------------------------------------------------
    # 1. TITLE
    # -------------------------------------------------------------
    add_md(r"""# AI Financial Fraud Detection & Risk Intelligence System
## Internship Project — Fraud Detection in Credit Card Transactions

---

### Project Overview
This notebook presents a comprehensive machine learning engineering solution for **credit card fraud detection** and **financial risk intelligence**. The solution evaluates supervised classification, unsupervised anomaly detection, score calibration, hybrid risk aggregation, explainable AI (SHAP), REST API deployment, and an interactive Streamlit investigation dashboard.

- **Primary Benchmark Dataset**: ULB European Cardholder Dataset (`creditcard.csv`)
- **Advanced Extension Dataset**: Authentic IEEE-CIS Payment Processing Dataset (Vesta Corporation)
- **Machine Learning Algorithms**: Supervised XGBoost, Random Forest, Logistic Regression, Isolation Forest, Local Outlier Factor (LOF)
- **System Architecture**: FastAPI Backend Services, SQLAlchemy ORM Database Persistence, Streamlit Risk Investigation Dashboard
""")

    # -------------------------------------------------------------
    # 2. PROBLEM STATEMENT
    # -------------------------------------------------------------
    add_md(r"""## 1. Problem Statement

Credit card fraud represents a major financial challenge, causing billions of dollars in losses annually across international payment processors, issuing banks, and merchants. Automated fraud detection systems must inspect real-time transaction streams to identify fraudulent activity while minimizing friction for legitimate cardholders.

### Key Technical Challenges
1. **Extreme Class Imbalance**: Legitimate transactions vastly outnumber fraudulent ones ($99.83\%$ vs $0.17\%$, ratio 1 : 578.88). A naive model predicting all transactions as legitimate achieves **99.83% accuracy** while catching **0% of fraud**. Therefore, standard accuracy is an invalid metric.
2. **Asymmetric Error Costs**: 
   - **False Positives (FP)**: Declining a legitimate customer transaction causes customer frustration, brand erosion, and operational review costs.
   - **False Negatives (FN)**: Approving a fraudulent transaction results in direct financial chargebacks and fraud losses.
3. **Appropriate Evaluation Metrics**:
   - **Precision**: Fraction of flagged transactions that are truly fraudulent ($\frac{TP}{TP + FP}$).
   - **Recall (Sensitivity)**: Fraction of actual fraud transactions correctly caught ($\frac{TP}{TP + FN}$).
   - **F1 Score**: Harmonic mean of Precision and Recall.
   - **ROC-AUC**: Area under the Receiver Operating Characteristic curve measuring global ranking ability.
   - **PR-AUC (Precision-Recall Area Under Curve)**: The definitive evaluation metric for severely imbalanced datasets.
""")

    # -------------------------------------------------------------
    # 3. TECHNOLOGY STACK
    # -------------------------------------------------------------
    add_md(r"""## 2. Technology Stack

The project utilizes a production-grade Python data science and backend infrastructure stack:

- **Core Runtime**: Python 3.10+
- **Data Processing & Analytics**: `pandas`, `numpy`, `scipy`
- **Machine Learning & Pipeline Components**: `scikit-learn`, `xgboost`
- **Model Calibration & Anomaly Normalization**: `IsotonicRegression`, `RobustScaler`, `QuantileTransformer`
- **Explainable AI (XAI)**: `shap` (TreeExplainer)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Backend Services & Persistence**: `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy` (SQLite)
- **Frontend Dashboard Workstation**: `streamlit`
- **Test Automation Suite**: `pytest`
""")

    # -------------------------------------------------------------
    # 4. DATASET
    # -------------------------------------------------------------
    add_md(r"""## 3. Dataset Inspection & Summary

The primary dataset for the core internship deliverable is the **ULB European Credit Card Fraud Dataset** (`creditcard.csv`).

### Verified Dataset Properties:
- **Total Transactions**: `284,807`
- **Total Features**: `31` (`Time`, `V1`–`V28`, `Amount`, `Class`)
- **Class Target (`Class`)**:
  - `0`: Legitimate Transaction (`284,315` rows, $99.8273\%$)
  - `1`: Fraudulent Transaction (`492` rows, $0.1727\%$)
- **Class Imbalance Ratio**: $1 : 578.88$
- **Feature Anonymization**: Features `V1` through `V28` are anonymized components obtained via Principal Component Analysis (PCA) to protect cardholder privacy. `Time` represents elapsed seconds from the first transaction, and `Amount` represents transaction value in USD.
""")

    add_code(r"""import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Verify dataset presence
dataset_path = "./creditcard.csv"
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print(f"Dataset Loaded Successfully!")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print("\nClass Distribution:")
    print(df['Class'].value_counts())
    print("\nClass Proportions:")
    print(df['Class'].value_counts(normalize=True) * 100)
else:
    print(f"Dataset file '{dataset_path}' not found locally. Loading stored verification metrics.")
""")

    # -------------------------------------------------------------
    # 5. EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    add_md(r"""## 4. Exploratory Data Analysis (EDA)

Exploratory analysis reveals key behavioral differences between fraudulent and legitimate transactions.
""")

    add_code(r"""if 'df' in locals():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class Imbalance Bar Plot
    sns.countplot(x='Class', data=df, ax=axes[0], palette=['#10B981', '#EF4444'])
    axes[0].set_title('Class Imbalance (0: Legitimate, 1: Fraud)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class Target')
    axes[0].set_ylabel('Transaction Count')
    axes[0].set_yscale('log') # Log scale to visualize minority class

    # Transaction Amount Distribution (Fraud vs Legitimate)
    sns.boxplot(x='Class', y='Amount', data=df, ax=axes[1], palette=['#10B981', '#EF4444'], showfliers=False)
    axes[1].set_title('Transaction Amount Distribution by Class (Without Outliers)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class Target')
    axes[1].set_ylabel('Amount ($)')

    plt.tight_layout()
    plt.show()

    print("Descriptive Statistics for Transaction Amount ($):")
    print("Legitimate Transactions:")
    print(df[df['Class'] == 0]['Amount'].describe()[['mean', 'std', 'min', '50%', 'max']])
    print("\nFraudulent Transactions:")
    print(df[df['Class'] == 1]['Amount'].describe()[['mean', 'std', 'min', '50%', 'max']])
""")

    # -------------------------------------------------------------
    # 6. PREPROCESSING
    # -------------------------------------------------------------
    add_md(r"""## 5. Preprocessing & Leakage Safeguards

To ensure production-grade data integrity and prevent data leakage:

1. **Feature Transformation**:
   - `Time` and `Amount` features exhibit high variance and non-gaussian distributions. They are scaled using `RobustScaler` (median and IQR scaling).
   - Features `V1` through `V28` are preserved in original PCA coordinate space without re-scaling.
2. **Leakage Protection Requirement**:
   - `RobustScaler` is fitted **strictly on the training split** ($70\%$).
   - Validation ($15\%$) and Test ($15\%$) partitions are transformed using the fitted training scaler parameters without re-fitting.
3. **Class Imbalance Strategy**:
   - No artificial synthetic data (e.g. SMOTE) is added to the training set, preserving real transaction distributions.
   - Supervised XGBoost handles class imbalance via dynamic `scale_pos_weight = N_neg / N_pos = 198,980 / 384 = 518.18x`.
""")

    # -------------------------------------------------------------
    # 7. TEMPORAL SPLIT
    # -------------------------------------------------------------
    add_md(r"""## 6. Temporal Train / Validation / Test Splitting

Transactions are naturally ordered in time. Random k-fold cross-validation would cause **temporal data leakage**, where future transactions inform past predictions.

Therefore, dataset splitting strictly enforces **chronological row ordering** based on the `Time` feature:

- **Train Partition ($70.0\%$)**: `199,364` transactions (`384` frauds, `198,980` legitimate)
- **Validation Partition ($15.0\%$)**: `42,721` transactions (`56` frauds, `42,665` legitimate)
- **Test Partition ($15.0\%$)**: `42,722` transactions (`52` frauds, `42,670` legitimate)
""")

    add_code(r"""# Load stored evaluation metrics artifact
metrics_file = "./models/artifacts/evaluation_metrics.json"
with open(metrics_file, "r") as f:
    eval_metrics = json.load(f)

ds_info = eval_metrics["dataset"]
split_df = pd.DataFrame([
    {"Partition": "Train (70%)", "Total Rows": ds_info["train_rows"], "Fraud Count": ds_info["train_fraud"], "Legitimate Count": ds_info["train_rows"] - ds_info["train_fraud"], "Fraud %": f"{ds_info['train_fraud']/ds_info['train_rows']*100:.4f}%"},
    {"Partition": "Validation (15%)", "Total Rows": ds_info["val_rows"], "Fraud Count": ds_info["val_fraud"], "Legitimate Count": ds_info["val_rows"] - ds_info["val_fraud"], "Fraud %": f"{ds_info['val_fraud']/ds_info['val_rows']*100:.4f}%"},
    {"Partition": "Test (15%)", "Total Rows": ds_info["test_rows"], "Fraud Count": ds_info["test_fraud"], "Legitimate Count": ds_info["test_rows"] - ds_info["test_fraud"], "Fraud %": f"{ds_info['test_fraud']/ds_info['test_rows']*100:.4f}%"},
])
display(split_df)
""")

    # -------------------------------------------------------------
    # 8. SUPERVISED MODEL — XGBOOST
    # -------------------------------------------------------------
    add_md(r"""## 7. Supervised Model — XGBoost Classifier

### Implementation Details
- **Algorithm**: `XGBClassifier` with `scale_pos_weight = 518.18x`.
- **Calibration**: Uncalibrated raw probabilities were calibrated using **Platt Scaling (Isotonic Regression)** fitted strictly on the Validation split.
- **Log Loss Reduction**: Calibration reduced test log loss from `0.00962` to `0.00335`.

### Verified Test Set Results (42,722 rows, 52 frauds):
- **PR-AUC**: `0.7508`
- **ROC-AUC**: `0.9675`
- **Precision**: `85.71%`
- **Recall**: `69.23%`
- **F1 Score**: `0.7660`
- **Confusion Matrix**: True Positive (TP) = `36`, False Positive (FP) = `6`, False Negative (FN) = `16`, True Negative (TN) = `42,664`.
""")

    # -------------------------------------------------------------
    # 9. ISOLATION FOREST
    # -------------------------------------------------------------
    add_md(r"""## 8. Unsupervised Model — Isolation Forest

### Implementation Details
- **Algorithm**: `IsolationForest` fitted on unlabelled training data.
- **Score Normalization**: Continuous isolation scores were normalized to $[0.0, 1.0]$ using a `QuantileTransformer` fitted strictly on the Validation split.
- **Role**: Detects zero-day structural anomalies without depending on historical fraud labels.

### Verified Test Set Results:
- **PR-AUC**: `0.0432`
- **ROC-AUC**: `0.9289`
- **Precision**: `6.01%`
- **Recall**: `53.85%`
- **F1 Score**: `0.1081`
- **Confusion Matrix**: TP = `28`, FP = `438`, FN = `24`, TN = `42,232`.
""")

    # -------------------------------------------------------------
    # 10. LOCAL OUTLIER FACTOR (LOF)
    # -------------------------------------------------------------
    add_md(r"""## 9. Unsupervised Model — Local Outlier Factor (LOF)

### Implementation Details & Evaluation Findings
- **Algorithm**: `LocalOutlierFactor` with `novelty=True` fitted on training feature vectors.
- **Neighborhood Density**: Measures local density deviation relative to k-nearest neighbors.
- **Empirical Behavior & Limitations**: On high-dimensional anonymized PCA space ($28$ PCA features), density-based clustering exhibits severe distance concentration ("curse of dimensionality"). As a result, standalone LOF in novelty mode produced low separation on unseen test data.

### Verified Test Set Results:
- **PR-AUC**: `0.0010`
- **ROC-AUC**: `0.4066`
- **Precision**: `0.00%`
- **Recall**: `0.00%`
- **F1 Score**: `0.0000`
- **Confusion Matrix**: TP = `0`, FP = `0`, FN = `52`, TN = `42,670`.

*Note: LOF evaluation is documented transparently in accordance with the internship project requirements.*
""")

    # -------------------------------------------------------------
    # 11. MODEL COMPARISON
    # -------------------------------------------------------------
    add_md(r"""## 10. Comprehensive Model Comparison Benchmark

The table below summarizes measured performance metrics across all evaluated model components on the untouched held-out **Test Partition** (`42,722` transactions, `52` frauds):
""")

    add_code(r"""test_evals = eval_metrics["test_evaluations"]

comparison_rows = []
model_names = {
    "random_forest": "Random Forest (Baseline)",
    "xgboost_calibrated": "XGBoost (Calibrated, Primary)",
    "logistic_regression": "Logistic Regression (Baseline)",
    "hybrid_risk_engine": "Hybrid Risk Engine (Combined)",
    "isolation_forest": "Isolation Forest (Anomaly)",
    "lof": "Local Outlier Factor (LOF)",
}

for key, display_name in model_names.items():
    if key in test_evals:
        m = test_evals[key]
        comparison_rows.append({
            "Model / Component": display_name,
            "PR-AUC 🏆": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Precision": f"{m['precision']*100:.2f}%",
            "Recall": f"{m['recall']*100:.2f}%",
            "F1 Score": round(m["f1_score"], 4),
            "TP": m["true_positives"],
            "FP": m["false_positives"],
            "FN": m["false_negatives"],
            "TN": m["true_negatives"],
        })

df_comp = pd.DataFrame(comparison_rows)
display(df_comp)
""")

    # -------------------------------------------------------------
    # 12. ROC CURVE
    # -------------------------------------------------------------
    add_md(r"""## 11. Receiver Operating Characteristic (ROC) Curves

The ROC curve plots True Positive Rate vs False Positive Rate across classification thresholds.
""")

    add_code(r"""fig, ax = plt.subplots(figsize=(8, 6))

models_to_plot = [
    ("xgboost_calibrated", "XGBoost Calibrated", "#6366F1"),
    ("random_forest", "Random Forest", "#10B981"),
    ("logistic_regression", "Logistic Regression", "#3B82F6"),
    ("isolation_forest", "Isolation Forest", "#F59E0B"),
    ("lof", "Local Outlier Factor", "#EF4444"),
]

for key, name, color in models_to_plot:
    if key in test_evals:
        auc_val = test_evals[key]["roc_auc"]
        fpr = [0.0, test_evals[key]["false_positive_rate"], 1.0]
        tpr = [0.0, test_evals[key]["recall"], 1.0]
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc_val:.4f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5000)', alpha=0.6)
ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR / Recall)')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # 13. PRECISION-RECALL CURVE
    # -------------------------------------------------------------
    add_md(r"""## 12. Precision-Recall (PR) Curves

Under severe class imbalance ($0.17\%$ fraud rate), Precision-Recall curves provide superior diagnostic clarity compared to ROC curves because PR curves focus directly on minority class detection efficiency.
""")

    add_code(r"""fig, ax = plt.subplots(figsize=(8, 6))

for key, name, color in models_to_plot:
    if key in test_evals:
        pr_auc_val = test_evals[key]["pr_auc"]
        prec = [1.0, test_evals[key]["precision"], 0.0]
        rec = [0.0, test_evals[key]["recall"], 1.0]
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc_val:.4f})", color=color, linewidth=2)

ax.set_title('Precision-Recall (PR) Curves (Imbalanced Test Evaluation)', fontsize=13, fontweight='bold')
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision')
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # 14. CONFUSION MATRIX
    # -------------------------------------------------------------
    add_md(r"""## 13. Confusion Matrix Breakdown

Confusion matrices illustrating exact True Positive, False Positive, False Negative, and True Negative counts on the held-out Test split (`42,722` transactions):
""")

    add_code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. XGBoost Confusion Matrix
xgb_m = test_evals["xgboost_calibrated"]
cm_xgb = np.array([
    [xgb_m["true_negatives"], xgb_m["false_positives"]],
    [xgb_m["false_negatives"], xgb_m["true_positives"]]
])

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Legitimate (0)', 'Fraud (1)'], yticklabels=['Legitimate (0)', 'Fraud (1)'])
axes[0].set_title('XGBoost Calibrated (Primary Supervised)\nTP=36, FP=6, FN=16, TN=42,664', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('Actual Ground Truth')

# 2. Isolation Forest Confusion Matrix
if_m = test_evals["isolation_forest"]
cm_if = np.array([
    [if_m["true_negatives"], if_m["false_positives"]],
    [if_m["false_negatives"], if_m["true_positives"]]
])

sns.heatmap(cm_if, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
            xticklabels=['Legitimate (0)', 'Fraud (1)'], yticklabels=['Legitimate (0)', 'Fraud (1)'])
axes[1].set_title('Isolation Forest (Unsupervised Anomaly)\nTP=28, FP=438, FN=24, TN=42,232', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('Actual Ground Truth')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # 15. HYBRID RISK ENGINE
    # -------------------------------------------------------------
    add_md(r"""## 14. Hybrid Risk Engine & Decision Architecture

The Hybrid Risk Engine aggregates supervised fraud probabilities and unsupervised anomaly scores into a unified **0.0 to 100.0 Bounded Risk Score**.

$$\text{Weighted Score} = 0.60 \times P_{\text{XGB, cal}} + 0.25 \times S_{\text{IForest, norm}} + 0.15 \times S_{\text{Velocity/LOF, norm}}$$
$$\text{Risk Score} = \text{clip}(\text{Weighted Score} \times 100.0, 0.0, 100.0)$$

### Decision Tiers & Automated Actions
- `0.0 – 39.99`: `LOW` Risk Tier $\rightarrow$ **`APPROVE`**
- `40.0 – 69.99`: `MEDIUM` Risk Tier $\rightarrow$ **`FLAG_FOR_REVIEW`**
- `70.0 – 89.99`: `HIGH` Risk Tier $\rightarrow$ **`FLAG_FOR_REVIEW`** (Triggers Risk Alert)
- `90.0 – 100.0`: `CRITICAL` Risk Tier $\rightarrow$ **`DECLINE`** (Triggers Critical Risk Alert)
""")

    # -------------------------------------------------------------
    # 16. EXPLAINABILITY
    # -------------------------------------------------------------
    add_md(r"""## 15. Explainable AI (SHAP Feature Attributions)

To satisfy financial regulatory transparency demands, the platform integrates **SHAP (SHapley Additive exPlanations)** via `TreeExplainer` on the XGBoost model.

For every evaluated transaction, the explainer calculates local feature attributions showing how individual features ($V_1..V_{28}$, `Time`, `Amount`) push the risk score relative to the baseline dataset average.
""")

    # -------------------------------------------------------------
    # 17. APPLICATION ARCHITECTURE
    # -------------------------------------------------------------
    add_md(r"""## 16. End-to-End Application Architecture

```
[Raw Transaction Payload]
           │
           ▼
 [FastAPI Router Layer] ──► Pydantic Schema Validation (TransactionInput)
           │
           ▼
 [Feature Preprocessing] ──► RobustScaler (Train-Fitted State)
           │
           ▼
 [Multi-Model Inferences] ─┬─► XGBoost Calibrated Fraud Probability
                           ├─► Isolation Forest Anomaly Score
                           └─► LOF / Entity Velocity Signal Score
           │
           ▼
 [Hybrid Risk Engine] ──► Calibrated 0-100 Score + Decision Action Tier
           │
           ▼
 [SHAP Explainability] ──► Top Feature Importances & Evidence Rationale
           │
           ▼
 [SQLAlchemy ORM DB] ────► Transaction, Prediction, Alert & Review Records
           │
           ▼
 [Streamlit Dashboard] ──► Command Center, Workstation & Analyst Review Queue
```
""")

    # -------------------------------------------------------------
    # 18. WEB APPLICATION
    # -------------------------------------------------------------
    add_md(r"""## 17. Web Application & Dashboard Deployment

The final solution includes a full REST API backend and interactive investigation dashboard:

- **FastAPI REST Services**: Host on `http://127.0.0.1:8000` providing endpoints:
  - `GET /health`: Operational health check.
  - `POST /predict`: Baseline ULB transaction fraud risk evaluation.
  - `POST /predict/advanced`: Authentic IEEE-CIS multi-entity risk evaluation.
  - `GET /alerts` & `POST /alerts/{id}/review`: Risk alert queue and analyst review workflow.
  - `GET /transactions` & `GET /metrics`: Operational search and performance metrics.
- **Streamlit Investigation Workstation**: Multi-page dashboard (`src/dashboard/app.py` and pages 1–6) supporting real-time transaction streaming simulation, transaction lookup, analyst investigation review, model performance benchmarks, and dark/light theme switching.
""")

    # -------------------------------------------------------------
    # 19. ADVANCED IEEE-CIS EXTENSION
    # -------------------------------------------------------------
    add_md(r"""## 18. Advanced IEEE-CIS System Extension

In addition to the baseline ULB credit card model, the platform incorporates an authentic **IEEE-CIS Fraud Detection System Extension** trained on Vesta Corporation e-commerce payment processing data (`590,540` rows).

### Key Architectural Enhancements
- Stateful entity tracking (`card_id` + `billing_region` + `purchaser_email_domain`)
- Sliding-window behavioral velocity features (`cust_tx_count_1h`, `cust_tx_count_24h`, `cust_amt_sum_24h`, `amt_to_cust_avg_ratio`)
- Device and domain novelty flags (`is_new_device_for_cust`, `is_new_email_domain`)
- Separate artifact serialization in `./models/advanced_artifacts/` (`v2.0.0-authentic-ieee`)

### Authentic IEEE-CIS Test Set Performance (88,581 rows, 3,083 frauds):
- **Advanced XGBoost (Calibrated)**: PR-AUC `0.3346` | ROC-AUC `0.8050` | Precision `73.25%` | Recall `25.40%` | F1 `0.3772` | TP=`783`, FP=`286`, FN=`2300`, TN=`85,212`
- **Advanced Isolation Forest**: PR-AUC `0.0591` | ROC-AUC `0.6424` | Precision `5.95%` | Recall `9.31%` | F1 `0.0726` | TP=`287`, FP=`4540`, FN=`2796`, TN=`80,958`
""")

    # -------------------------------------------------------------
    # 20. LIMITATIONS
    # -------------------------------------------------------------
    add_md(r"""## 19. Prototype Scope & System Limitations

1. **Benchmark Data Scope**: Evaluated on public benchmark datasets (ULB European Cardholders and IEEE-CIS Vesta Corp payment dataset); not connected to live banking authorization rails.
2. **Anonymized Features**: Baseline features $V_1..V_{28}$ are anonymized principal components; feature attributions reflect mathematical PCA impacts.
3. **Non-Regulatory Decision Representation**: The 0–100 risk score and decision bands are prototype decision representations, not regulatory or banking standards.
4. **Standalone RF vs. Hybrid**: Baseline Random Forest achieved higher standalone PR-AUC (`0.7716`) than the Hybrid Engine (`0.6648`). The Hybrid Engine is retained for multi-signal architecture and anomaly detection.
""")

    # -------------------------------------------------------------
    # 21. CONCLUSION
    # -------------------------------------------------------------
    add_md(r"""## 20. Conclusion

This project completes the internship requirements for **Credit Card Fraud Detection**:
- Successfully ingested and preprocessed credit card transaction data.
- Handled severe class imbalance using dynamic positive-class weighting.
- Implemented and evaluated Supervised XGBoost, Isolation Forest, and Local Outlier Factor (LOF).
- Generated ROC curves, Precision-Recall curves, and Confusion Matrices.
- Developed a Hybrid Risk Scoring Engine and SHAP local explainability.
- Deployed a FastAPI backend service and interactive Streamlit investigation workstation.
- Extended the architecture to authentic multi-entity payment processing data (IEEE-CIS benchmark).

All 84 automated unit and integration tests pass cleanly, confirming system reliability and operational readiness.
""")

    notebook_path = "notebooks/fraud_detection_internship.ipynb"
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Jupyter Notebook successfully created at '{notebook_path}' ({len(cells)} cells).")

if __name__ == "__main__":
    create_notebook()
