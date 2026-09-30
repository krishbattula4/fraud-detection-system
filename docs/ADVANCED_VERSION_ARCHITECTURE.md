# Advanced Version System Architecture & Implementation Blueprint
**AI Financial Fraud Detection & Risk Intelligence Platform**

---

## 1. Executive Summary & Architectural Goals

The Advanced System extends the baseline ULB prototype into an enterprise-class **Multi-Model Fraud Risk & Behavioral Intelligence Platform** powered by real payment processing data (IEEE-CIS Fraud Detection Benchmark schema).

### Core Architectural Guarantees:
1. **100% Baseline Preservation:** The existing ULB Kaggle model (`creditcard.csv`), schemas (`TransactionInput`), saved joblib artifacts (`models/artifacts/`), and API endpoints remain fully functional and untouched as an empirical benchmark.
2. **Advanced Multi-Entity Fraud Intelligence:** Introduces real-world financial transaction concepts:
   - Stateful customer entity tracking (`card1` + `addr1` + `P_emaildomain`)
   - Leakage-safe sliding-window behavioral velocity features (`cust_tx_count_1h`, `cust_tx_count_24h`, `cust_amt_sum_24h`, `amt_to_cust_avg_ratio`)
   - Device attributes (`DeviceType`, `DeviceInfo`, OS, Browser)
   - Category attributes (`ProductCD`, `card4` network, `card6` debit/credit)
   - Multi-layer decisioning combining Supervised ML, Unsupervised Anomaly Detection, and Deterministic Risk Rules.
3. **Artifact Isolation:** Advanced model artifacts are serialized strictly to `models/advanced_artifacts/`, preventing baseline overwrite risks.

---

## 2. End-to-End System Pipeline Diagram

```
                             TRANSACTION INGESTION
                                       │
                                       ▼
                       Schema & Range Validation
                   (Pydantic: AdvancedTransactionInput)
                                       │
                                       ▼
                     DataPreprocessor & Robust Scaler
                                       │
                                       ▼
                    Stateful Behavioral Feature Engine
           (Sliding-Window Velocity & Device/Domain Novelty)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   Calibrated XGBoost          Isolation Forest            Behavioral Rules
 (Supervised Fraud Prob)       (Anomaly Detector)         (Velocity & Device)
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                         Advanced Hybrid Risk Engine
                     (Weighted Ensemble + Rule Override)
                                       │
                                       ▼
                        Calibrated 0-100 Risk Score
                                       │
                                       ▼
                    Risk Tier & Automated Decision Action
           (LOW / MEDIUM / HIGH / CRITICAL ➔ APPROVE/FLAG/DECLINE)
                                       │
                                       ▼
                      Multi-Layer Explainability
         (SHAP Attributions + Behavioral Rationale + Rule Signals)
                                       │
                                       ▼
                     SQLAlchemy ORM Persistence
                                       │
                                       ▼
                 FastAPI REST API / Streamlit Dashboard
```

---

## 3. Detailed Component Architecture

### A. Data Ingestion & Validation (`src/schemas/advanced_transaction.py`)
- Defines `AdvancedTransactionInput` validation schema:
  - `transaction_id`: String (UUID or client ID)
  - `timestamp`: Double (Elapsed time in seconds `TransactionDT` or epoch)
  - `amount`: Float (`ge=0.0`)
  - `card_id`: String (e.g. `card1` issuer code)
  - `billing_region`: Optional[String] (`addr1`)
  - `purchaser_email_domain`: Optional[String] (`P_emaildomain`)
  - `recipient_email_domain`: Optional[String] (`R_emaildomain`)
  - `product_category`: Optional[String] (`ProductCD`: W, C, R, H, S)
  - `card_network`: Optional[String] (`card4`: visa, mastercard, etc.)
  - `card_type`: Optional[String] (`card6`: credit, debit)
  - `device_type`: Optional[String] (`DeviceType`: desktop, mobile)
  - `device_info`: Optional[String] (`DeviceInfo`: iOS, Android, Windows, etc.)
  - `counting_features`: Dict (`C1`–`C14`)
  - `timedelta_features`: Dict (`D1`–`D15`)

### B. Stateful Behavioral Feature Engine (`src/data/advanced_preprocessing.py`)
- Maintains chronological state over past transactions (`t_prev < t_current`).
- Computes sliding window aggregations:
  - `cust_tx_count_1h`: Transactions in 3,600s window.
  - `cust_tx_count_24h`: Transactions in 86,400s window.
  - `cust_amt_sum_24h`: Sum of transaction amounts in 86,400s window.
  - `amt_to_cust_avg_ratio`: Amount divided by customer 30-day average (`cust_amt_avg_30d`).
  - `is_new_device_for_cust`: Boolean flag indicating unseen device for entity.
  - `is_new_email_domain`: Boolean flag indicating unseen recipient domain.

### C. ML & Anomaly Models (`src/ml/advanced_models.py`)
- **Advanced Calibrated XGBoost (`AdvancedXGBoostModel`)**: Supervised model trained on IEEE-CIS transaction features, calibrated via Isotonic Regression on Validation split.
- **Advanced Isolation Forest (`AdvancedIsolationForest`)**: Global anomaly tree model fitted on unlabelled transaction vectors.

### D. Advanced Hybrid Risk Engine (`src/engine/advanced_risk_engine.py`)
- Aggregates model signals and applies deterministic risk safeguards:
  - **Supervised Weight:** $0.70$
  - **Anomaly Weight:** $0.15$
  - **Behavioral Velocity Weight:** $0.15$
  - **Rule Safeguards:**
    * If `cust_tx_count_1h > 10` ➔ Force minimum Risk Score $\ge 75.0$ (`HIGH`).
    * If `amt_to_cust_avg_ratio > 5.0` and `is_new_device_for_cust == True` ➔ Force minimum Risk Score $\ge 85.0$ (`HIGH`).
- Output: 0–100 bounded score, risk tier (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and decision (`APPROVE`, `FLAG_FOR_REVIEW`, `DECLINE`).

### E. Explainability Engine (`src/explainability/advanced_explainer.py`)
- Combines:
  1. SHAP TreeExplainer feature attributions for XGBoost model signals.
  2. Behavioral velocity alerts (e.g. "Transaction amount is 4.2x above customer 30-day average").
  3. Rule override explanations when deterministic safeguards trigger.

### F. Persistence Layer (`src/db/models.py` & `repository.py`)
- Stores advanced predictions in `prediction_records` table with model version tag `v2.0.0-advanced`.
- Stores alerts and analyst reviews cleanly via SQLAlchemy ORM.

### G. API Contracts (`src/api/routes.py`)
- `POST /predict/advanced`: Real-time advanced transaction risk scoring.
- `POST /predict`: Existing ULB baseline endpoint (preserved 100%).
- `GET /metrics/advanced`: Advanced engine performance and risk tier stats.

### H. Dashboard Workstation (`src/dashboard/pages/5_Simulator.py`)
- Adds IEEE-CIS dataset selector with historical label preview (`isFraud=1` / `isFraud=0` shown for context, NOT sent into prediction).
- Advanced technical input form for custom payload testing.

---

## 4. Migration & File Ownership Plan

| Action | Target Path | Purpose |
| :--- | :--- | :--- |
| **UNTOUCHED** | `src/data/preprocessing.py` | Baseline ULB RobustScaler preprocessor |
| **UNTOUCHED** | `models/artifacts/*` | All baseline joblib model artifacts |
| **UNTOUCHED** | `src/engine/risk_engine.py` | Baseline ULB Risk Engine |
| **UNTOUCHED** | `tests/unit/*` & `tests/integration/*` | All existing 67 baseline test cases |
| **CREATE** | `src/schemas/advanced_transaction.py` | Pydantic schema for advanced transaction vector |
| **CREATE** | `src/data/advanced_preprocessing.py` | Stateful sliding-window behavioral velocity engine |
| **CREATE** | `models/advanced_artifacts/` | Directory for version 2.0.0 advanced artifacts |
| **CREATE** | `src/ml/advanced_models.py` | Advanced XGBoost & Isolation Forest wrappers |
| **CREATE** | `src/engine/advanced_risk_engine.py` | Advanced Hybrid Risk Engine with rule safeguards |
| **CREATE** | `src/explainability/advanced_explainer.py` | Multi-layer SHAP & behavioral rationale generator |
| **MODIFY** | `src/api/routes.py` | Non-breaking `/predict/advanced` endpoint addition |
| **MODIFY** | `src/dashboard/pages/5_Simulator.py` | IEEE-CIS simulator selector & advanced mode |

---

## 5. Verification Plan

1. **Unit & Pipeline Tests:**
   - Execute baseline pytest suite (`python -m pytest tests/ -v`). Confirm 67/67 pass.
   - Write unit tests in `tests/unit/test_advanced_pipeline.py` verifying stateful sliding window aggregations, zero future data leakage, and risk engine output bounds.
2. **API Endpoint Verification:**
   - Verify `POST /predict` (Baseline ULB) returns 200 OK.
   - Verify `POST /predict/advanced` (IEEE-CIS Engine) returns 200 OK with calibrated 0–100 score and behavioral explanations.
3. **Dashboard Verification:**
   - Verify Streamlit Command Center, Transactions, Alerts, Simulator, Model Intelligence, and System Health in both Dark and Light theme modes.

---

**STOP GATE 2 REACHED — WAITING FOR USER APPROVAL OF ARCHITECTURE BLUEPRINT BEFORE IMPLEMENTATION.**
