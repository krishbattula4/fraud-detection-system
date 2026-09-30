# AI Financial Fraud Detection & Risk Intelligence System
## Final System Status & Technical Documentation

### 1. System Purpose
This repository implements a production-style, multi-model credit card fraud detection and risk intelligence system evaluated on the real ULB Kaggle credit-card dataset (`creditcard.csv`). The architecture combines supervised machine learning probability scoring, unsupervised multi-algorithm anomaly detection, probability calibration, score normalization, transaction-level SHAP feature attribution, automated decisioning workflows, database persistence, REST API services, and a multi-page Streamlit investigation dashboard.

---

### 2. Pipeline Architecture
```
[Credit Card Transaction]
         ↓
[Pydantic Schema & Feature Validation]
         ↓
[DataPreprocessor: RobustScaler on Time & Amount (Train-Fitted)]
         ↓
 ┌─────────────────────────────────────────────────────────────┐
 │               Hybrid Risk Detection Engine                  │
 │  - Supervised Model (XGBoost, scale_pos_weight=518.18x)     │
 │  - Global Anomaly Detector (Isolation Forest)               │
 │  - Local Anomaly Detector (Local Outlier Factor, novelty)   │
 └─────────────────────────────────────────────────────────────┘
         ↓
[Platt Probability Calibration + MinMax Anomaly Normalization]
         ↓
[Weighted Risk Aggregation -> 0.0 - 100.0 Bounded Risk Score]
         ↓
[Risk Level Tier (LOW / MEDIUM / HIGH / CRITICAL) & Decision Action]
         ↓
[SHAP TreeExplainer Local Feature Attribution]
         ↓
[SQLAlchemy ORM Database Persistence (SQLite)]
         ↓
[FastAPI REST API / Streamlit Investigation Dashboard & Simulator]
```

---

### 3. Real Dataset Summary
- **Source File**: `creditcard.csv` (ULB European Cardholder Dataset, 2013).
- **Total Transactions**: `284,807`
- **Total Features**: `31` (`Time`, `V1`–`V28`, `Amount`, `Class`).
- **Target Distribution**: `284,315` legitimate transactions (`99.8273%`), `492` fraudulent transactions (`0.1727%`).
- **Class Imbalance Ratio**: 1 : 578.88.
- **Missing / Non-finite Values**: 0 missing values across all columns.

---

### 4. Chronological Data Partitioning
To prevent temporal data leakage, dataset splitting strictly enforced chronological order by `Time`:
- **Train Split (70.0%)**: `199,364` transactions (`384` frauds, `198,980` legitimate).
- **Validation Split (15.0%)**: `42,721` transactions (`56` frauds, `42,665` legitimate).
- **Test Split (15.0%)**: `42,722` transactions (`52` frauds, `42,670` legitimate).

---

### 5. Preprocessing & Leakage Protection
- **`DataPreprocessor`**: Fits `RobustScaler` on `Time` and `Amount` **strictly on the 70% Train split** (`199,364` rows).
- **PCA Features (`V1`–`V28`)**: Preserved in original PCA coordinate space without re-scaling.
- **Validation/Test**: Transformed using saved training scaler parameters (`time_scaler.center_ = [67232.0]`) without re-fitting.

---

### 6. Trained Supervised Models
1. **Logistic Regression (Baseline)**: Trained on `X_train`.
2. **Random Forest (Baseline)**: Trained on `X_train` (`n_estimators=100`, `random_state=42`).
3. **XGBoost (Primary Supervised)**: Trained on `X_train` with dynamic class balancing `scale_pos_weight = 198,980 / 384 = 518.18x`.

---

### 7. Probability Calibration
- **`ProbabilityCalibrator`**: Platt Scaling (Logistic Regression) fitted **strictly on 15% Validation split** probabilities (`uncal_val_probs`, `y_val`).
- **Output**: Well-calibrated probabilities in $[0.0, 1.0]$ with reduced log loss on unseen test data (`0.00335`).

---

### 8. Unsupervised Anomaly Detection & Normalization
- **Isolation Forest (`IsolationForestAnomalyDetector`)**: Global tree isolation model fitted on `X_train`. Score inverted (`-score_samples()`) so higher score = higher anomaly likelihood.
- **Local Outlier Factor (`LOFAnomalyDetector`)**: Local density model fitted on `X_train` with `novelty=True`.
- **`MinMaxScoreNormalizer`**: Score normalizers fitted **strictly on 15% Validation split anomaly scores** to map continuous raw scores bounded to $[0.0, 1.0]$.

---

### 9. Hybrid Risk Engine Configuration
- **Validation Search**: Grid search evaluated on Validation split PR-AUC selected optimal component weights:
  $$\text{Weight}_{\text{XGB}} = 0.80, \quad \text{Weight}_{\text{IForest}} = 0.10, \quad \text{Weight}_{\text{LOF}} = 0.10$$
- **Validation PR-AUC Achieved**: `0.7624`.
- **Score Scaling**: Aggregated score scaled to $[0.0, 100.0]$.

---

### 10. Risk Bands & Decision Action Mapping

| Risk Band Range | Risk Level Tier | Decision Action | Operational Action |
| :--- | :--- | :--- | :--- |
| `0.0 – 39.99` | `LOW` | `APPROVE` | Transaction approved automatically |
| `40.0 – 69.99` | `MEDIUM` | `FLAG_FOR_REVIEW` | Queued for analyst inspection |
| `70.0 – 89.99` | `HIGH` | `FLAG_FOR_REVIEW` | High priority alert triggered |
| `90.0 – 100.0` | `CRITICAL` | `DECLINE` | Transaction declined automatically |

*Note: Risk scores and bands are project-specific decision representations, not regulatory or banking standards.*

---

### 11. SHAP Local Transaction Attribution
- **`SHAPTransactionExplainer`**: TreeExplainer fitted on the XGBoost artifact using background sample data from `X_train`.
- **Feature Names**: Preserves exact dataset feature names (`V1`–`V28`, `Time`, `Amount`) without inventing arbitrary human labels.
- **Output**: Per-transaction top-$k$ feature importances returned in API responses and visual waterfall charts.

---

### 12. REST API Endpoints
- `POST /predict`: Real-time transaction fraud risk scoring, SHAP attribution, and persistence.
- `GET /alerts`: Retrieve flagged alerts with pagination and status filtering.
- `POST /alerts/{alert_id}/review`: Record analyst review decision and notes.
- `GET /transactions` & `GET /transactions/{id}`: Transaction search and detail lookup.
- `GET /metrics`: Operational evaluation metrics and risk band counts.
- `GET /health`: System health and model artifact readiness check (`model_artifacts: available`).

---

### 13. Persistence Layer
- **SQLAlchemy ORM**: Database models (`TransactionRecord`, `PredictionRecord`, `AlertRecord`, `ReviewRecord`).
- **Repositories**: Encapsulated data access pattern (`TransactionRepository`, `PredictionRepository`, `AlertRepository`, `ReviewRepository`, `MetricsRepository`).

---

### 14. Streamlit Multi-Page Dashboard
- `app.py`: Main entrypoint & sidebar backend status checker.
- `1_Overview.py`: Operational KPIs, risk-tier charts, recent alert queue.
- `2_Transactions.py`: Individual transaction lookup & SHAP attribution waterfall chart.
- `3_Alerts.py`: Analyst review & resolution workflow.
- `4_Simulator.py`: Interactive transaction streaming and single-transaction risk simulator.
- `5_System_Health.py`: Environment, database, and model artifact readiness monitoring.
- `4_Model_Performance.py`: Measured test set evaluation metrics, Plotly benchmark charts, confusion matrix breakdown, and disclosures.

---

### 15. Transaction Simulator
- Concrete `TransactionSimulator` in `src/simulation/simulator.py` preserving `BaseTransactionSimulator` contract (`stream_transactions`).
- Dispatches transactions directly to REST API (`POST /predict`), ensuring complete pipeline execution without duplicating ML logic.

---

### 16. Measured Test Set Evaluation Results

Evaluated on untouched Test split (`42,722` transactions, `52` fraud cases):

| Model / Component | PR-AUC 🏆 | ROC-AUC | Precision | Recall | F1 Score | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.7716** | 0.9608 | **95.12%** | 75.00% | **0.8387** | 39 | **2** | 13 |
| **XGBoost (Calibrated)** | **0.7508** | 0.9675 | 85.71% | 69.23% | 0.7660 | 36 | 6 | 16 |
| **Logistic Regression** | **0.7158** | **0.9773** | 5.77% | **82.69%** | 0.1079 | **43** | 702 | 9 |
| **Hybrid Risk Engine** | **0.6648** | 0.9399 | 85.71% | 69.23% | 0.7660 | 36 | 6 | 16 |
| **Isolation Forest** | **0.0432** | 0.9289 | 6.01% | 53.85% | 0.1081 | 28 | 438 | 24 |
| **Local Outlier Factor** | **0.0010** | 0.4066 | 0.00% | 0.00% | 0.0000 | 0 | 0 | 52 |

---

### 17. Prototype Scope & System Limitations
1. **PCA Anonymization**: Features `V1`–`V28` are anonymized principal components; SHAP feature names reflect raw PCA coordinates.
2. **Test Set Fraud Count**: Test evaluation split contains 52 real fraud transactions.
3. **LOF Anomaly Behavior**: LOF in novelty mode on un-clustered 30D PCA space exhibits limited separation (`PR-AUC 0.0010`).
4. **Standalone RF vs. Hybrid**: Random Forest achieved higher standalone PR-AUC (`0.7716`) than the Hybrid Engine (`0.6648`). The Hybrid Engine is retained for multi-signal architecture.
5. **Non-Regulatory**: 0–100 risk scores and decision bands are prototype decision representations, not banking standards.

---

### 18. Security & Privacy Safeguards
- Zero committed secrets or API keys.
- `.env` and `creditcard.csv` managed via `.gitignore`.
- Configuration managed cleanly via `pydantic-settings`.

---

### 19. Installation & Execution Guide

```powershell
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Automated Test Suite
python -m pytest tests/ -v

# 3. Start FastAPI REST Backend (Port 8000)
uvicorn src.api.app:app --reload --port 8000

# 4. Start Streamlit Dashboard (Port 8501)
streamlit run src/dashboard/app.py
```

---

### 20. Automated Test Suite Verification
- **Total Test Cases**: `67`
- **Pass Rate**: **100%** (`67 passed`, 0 failed).
- **Execution Time**: ~11.5 seconds.
