# Advanced Version Research & Dataset Selection Document
**AI Financial Fraud Detection & Risk Intelligence System**

---

## 1. Current System Audit

The current system is a fully functional, production-style fraud detection prototype evaluated on the ULB European credit card dataset (`creditcard.csv`).

### Current Architecture Summary:
- **Core Configuration & Logging:** `src/core/config.py`, `exceptions.py`, `logging.py` managing environment settings via Pydantic.
- **Data Pipeline:** `src/data/preprocessing.py` implementing `DataPreprocessor` with `RobustScaler` on `Time` and `Amount` fitted strictly on 70% Train split (`199,364` transactions).
- **ML Models:** `src/ml/supervised.py` (XGBoost with `scale_pos_weight=518.18x`, Random Forest, Logistic Regression), `src/ml/anomaly.py` (Isolation Forest, LOF with `novelty=True`), `src/ml/calibration.py` (Platt Scaling probability calibrator fitted on 15% Validation split).
- **Hybrid Risk Engine:** `src/engine/risk_engine.py` aggregating calibrated XGBoost probability ($0.80$), normalized Isolation Forest anomaly score ($0.10$), and normalized LOF anomaly score ($0.10$) into a calibrated $0.0 - 100.0$ risk score mapped to risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and automated decision actions (`APPROVE`, `FLAG_FOR_REVIEW`, `DECLINE`).
- **Explainability:** `src/explainability/explainer.py` (`SHAPTransactionExplainer`) using TreeExplainer for local feature attributions.
- **Persistence & API:** `src/db/repository.py` (SQLAlchemy ORM repositories), `src/api/app.py` (FastAPI REST API providing `POST /predict`, `GET /alerts`, `POST /alerts/{id}/review`, `GET /transactions`, `GET /metrics`, `GET /health`).
- **Frontend Command Center:** `src/dashboard/app.py` and Streamlit pages (`1_Overview.py` through `6_System_Health.py`) supporting dark/light theme switching, 3D WebGL network visualization (`pipeline_3d.py`), and live simulation (`5_Simulator.py`).

---

## 2. Current ULB Baseline Performance

Evaluated on untouched 15% Test split (`42,722` transactions, `52` real fraud cases):

| Model / Component | PR-AUC 🏆 | ROC-AUC | Precision | Recall | F1 Score | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Baseline)** | **0.7716** | 0.9608 | **95.12%** | 75.00% | **0.8387** | 39 | **2** | 13 |
| **XGBoost (Calibrated, Primary)** | **0.7508** | 0.9675 | 85.71% | 69.23% | 0.7660 | 36 | 6 | 16 |
| **Logistic Regression (Baseline)** | **0.7158** | **0.9773** | 5.77% | **82.69%** | 0.1079 | **43** | 702 | 9 |
| **Hybrid Risk Engine** | **0.6648** | 0.9399 | 85.71% | 69.23% | 0.7660 | 36 | 6 | 16 |
| **Isolation Forest (Anomaly)** | **0.0432** | 0.9289 | 6.01% | 53.85% | 0.1081 | 28 | 438 | 24 |
| **Local Outlier Factor (LOF)** | **0.0010** | 0.4066 | 0.00% | 0.00% | 0.0000 | 0 | 0 | 52 |

---

## 3. Current Limitations of ULB Baseline

1. **Anonymized PCA Features (`V1`–`V28`):**
   - The ULB dataset masks real financial transaction parameters into anonymized Principal Components (`V1` through `V28`).
   - While mathematical feature importance (SHAP) is accurate, domain-specific human explanations (e.g. "transaction amount is 5x above cardholder 30-day baseline" or "new device used in foreign location") cannot be derived without fabricating non-existent metadata.
2. **Lack of Entity Tracking:**
   - The ULB dataset contains no `customer_id`, `account_id`, `card_id`, `merchant_id`, or `device_id`.
   - As a result, stateful behavioral velocity features (e.g. number of transactions by customer in past 1 hour / 24 hours, velocity spike, new merchant category) cannot be computed.
3. **Static Single-Transaction View:**
   - The model evaluates each transaction in isolation without historical entity context.

---

## 4. Candidate Advanced Public Fraud Datasets

To upgrade the prototype into a realistic enterprise transaction-risk platform, three major public fraud detection datasets were researched:

### Candidate Dataset Evaluation Matrix:

| Evaluation Criteria | Candidate A: IEEE-CIS Fraud Detection (Kaggle / Vesta) | Candidate B: PaySim Synthetic Financial (Kaggle) | Candidate C: Sparkov Credit Card Fraud Simulation |
| :--- | :--- | :--- | :--- |
| **Data Source** | Real e-commerce payment transaction data (Vesta Corp) | Agent-based mobile money simulation | Synthetic bank transaction simulator |
| **Dataset Size** | 590,540 transactions (`train_transaction.csv` + `train_identity.csv`) | 6,362,620 transactions | 1,852,394 transactions |
| **Fraud Rate** | 3.50% (20,663 fraud cases out of 590,540) | 0.129% (8,213 fraud cases out of 6.36M) | 0.52% (9,651 fraud cases out of 1.85M) |
| **Temporal Structure** | `TransactionDT` (timedelta in seconds from reference start) | `step` (1 step = 1 hour) | `trans_date_trans_time` (real ISO timestamps) |
| **Entity Identifiers** | `card1`–`card6` (card details/issuer), `addr1`/`addr2` (billing region), `P_emaildomain`/`R_emaildomain`, `DeviceType`, `DeviceInfo` | `nameOrig` (customer ID), `nameDest` (recipient account ID) | `cc_num` (cardholder ID), `merchant`, `category`, `job`, `zip` |
| **Behavioral Velocity Potential** | High (can construct entity keys `card1` + `addr1` or `card1` + `P_emaildomain`) | High (customer ID `nameOrig` provided directly) | High (card number `cc_num` provided directly) |
| **Device & Location Signals** | Directly available (`DeviceType`, `DeviceInfo`, `addr1`, `addr2`, `dist1`, `dist2`, `id_30` OS, `id_31` browser) | Unavailable | Customer Lat/Long vs Merchant Lat/Long (distance calculable) |
| **Realism & Complexity** | Real production payment processor network data | Synthetic rule simulation | Synthetic rule simulation |

---

## 5. Feature Availability Analysis

### IEEE-CIS Fraud Detection Dataset (Recommended Candidate):

#### A. Fields Directly Available:
- **Transaction Core:** `TransactionID`, `TransactionDT` (seconds elapsed), `TransactionAmt` (amount in USD).
- **Payment & Card Info:** `ProductCD` (product code/category), `card1` (card ID/issuer code), `card2` (bank code), `card3` (country code), `card4` (card network: visa, mastercard, discover, etc.), `card5` (card type), `card6` (debit/credit).
- **Network & Location:** `addr1` (billing region), `addr2` (billing country), `dist1`/`dist2` (distances), `P_emaildomain` (purchaser email domain), `R_emaildomain` (recipient email domain).
- **Count Signals (`C1`–`C14`):** Counting features measuring transaction frequency across associated emails, cards, and addresses.
- **Time Delta Signals (`D1`–`D15`):** Days elapsed since previous transaction, card registration, or address change.
- **Match Signals (`M1`–`M9`):** Binary flags indicating match between name on card, billing address, and shipping address.
- **Device & Identity Attributes (`id_01`–`id_38`, `DeviceType`, `DeviceInfo`):** Device OS (iOS, Android, Windows), Browser version, screen resolution, connection type.
- **Target:** `isFraud` (0 = Legitimate, 1 = Fraudulent).

#### B. Fields Derivable Through Feature Engineering:
- **Customer Entity Key:** Constructed via `card1` + `addr1` + `P_emaildomain` to group transactions belonging to the same cardholder/account.
- **Behavioral Velocity:**
  * `cust_tx_count_1h`: Number of transactions by customer entity in past 1 hour.
  * `cust_tx_count_24h`: Number of transactions by customer entity in past 24 hours.
  * `cust_amt_sum_24h`: Total amount spent by customer entity in past 24 hours.
  * `amt_to_cust_avg_ratio`: Ratio of current transaction amount relative to customer's historical 30-day average amount.
- **Temporal Features:** `hour_of_day` (`(TransactionDT // 3600) % 24`), `day_of_week` (`(TransactionDT // 86400) % 7`).
- **Device & Domain Novelty:**
  * `is_new_device_for_cust`: Flag indicating whether device type/info has been seen before for this customer entity.
  * `is_new_email_domain`: Flag indicating first time seeing recipient email domain for this customer.

#### C. Fields Unavailable (Honest Disclosures):
- Raw cardholder full name, raw 16-digit credit card PAN, SSN, CVV (redacted for compliance/privacy).
- Live bank authorization response codes (ISO 8583 codes).

---

## 6. Feature Engineering & Leakage Prevention Strategy

To strictly prevent temporal data leakage during feature engineering:
1. **Sliding-Window Aggregations:** When computing velocity features (`cust_tx_count_24h`, `cust_amt_sum_24h`), only past transactions occurring strictly before current timestamp (`t_prev < t_current`) are included.
2. **Train-Only Parameter Fitting:**
   - Categorical frequency encoders, scalers, imputers, and baseline means are computed **strictly on the Training split**.
   - Validation and Test splits are transformed using stored training parameters without re-fitting.
3. **No Target Leakage:** `isFraud` is strictly excluded from all feature engineering calculations.

---

## 7. Recommended Advanced Dataset & Justification

### Recommended Dataset: **IEEE-CIS Fraud Detection Dataset** (Structured Subset)

### Justification:
1. **Real Production Data:** Derived from Vesta Corporation's real e-commerce payment gateway transactions, capturing real complex fraud attack patterns rather than synthetic simulation rules.
2. **Rich Entity & Behavioral Features:** Provides card attributes (`card1`–`card6`), device attributes (`DeviceType`, `DeviceInfo`), domain attributes, time deltas (`D1`–`D15`), and counting attributes (`C1`–`C14`).
3. **Enables Domain-Specific Risk Explanations:** Allows generating human-understandable risk explanations (e.g., "Transaction amount is 4.2x higher than customer 24h average", "Unusual device type used", "High 1h transaction velocity") alongside SHAP attributions.
4. **Preserves Baseline Coexistence:** The system architecture will support dual modes:
   - **Baseline Mode (ULB 28 PCA Features)**: Retained 100% functional as an empirical baseline benchmark.
   - **Advanced Mode (IEEE-CIS Multi-Signal Entity Features)**: Powers the advanced transaction risk intelligence platform!

---

## 8. Proposed Advanced System Architecture

```
                       ADVANCED TRANSACTION PAYLOAD
                                    │
                                    ▼
                      Input Schema & Validation
                                    │
                                    ▼
                      DataPreprocessor & Scaler
                                    │
                                    ▼
                Stateful Behavioral Feature Generator
           (Velocity, Amount Ratios, Device & Domain Novelty)
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    Supervised Model        Anomaly Detector       Behavioral Risk Rules
  (Calibrated XGBoost)      (Isolation Forest)      (Velocity & Device)
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                         Advanced Risk Engine
                (Weighted Ensemble + Rule Safeguards)
                                    │
                                    ▼
                     Calibrated 0-100 Risk Score
                                    │
                                    ▼
                       Risk Tier & Automated Decision
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

## 9. Migration & Extension Strategy

- **Baseline ULB Files (UNTOUCHED):**
  - `src/data/preprocessing.py`
  - `models/artifacts/` (Existing baseline joblib files)
  - `src/schemas/transaction.py` (Baseline `TransactionInput`)
- **Advanced Engine Files (NEW EXTENSIONS):**
  - `src/schemas/advanced_transaction.py` (Advanced transaction schema with entity/device attributes)
  - `src/data/advanced_preprocessing.py` (Advanced feature pipeline & entity velocity computation)
  - `src/ml/advanced_models.py` (Advanced XGBoost & Isolation Forest models)
  - `models/advanced_artifacts/` (Separate artifact storage directory preventing baseline overwrites)
  - `src/engine/advanced_risk_engine.py` (Advanced hybrid risk engine integrating ML + Anomaly + Behavioral Rules)

---

## 10. Verification Plan

1. **Unit Tests (`pytest`):**
   - Test advanced feature pipeline for zero future data leakage across sliding windows.
   - Test schema validation with valid, missing, and malformed inputs.
   - Test baseline regression to ensure existing 67 tests continue passing without regression.
2. **API Integration Tests:**
   - Test REST API prediction endpoints for both baseline and advanced payloads.
3. **Dashboard Verification:**
   - Test Streamlit Command Center, Transactions, Alerts, Simulator (Demo & Advanced modes), Model Intelligence, and System Health in both Dark and Light theme modes.

---

## 11. Stop Gate A Checklist & Status

- [x] Complete System Inspection performed across all codebase directories.
- [x] Existing baseline code, models, artifacts, and test suite audited.
- [x] Candidate public fraud datasets researched and evaluated.
- [x] Recommended dataset selected and justified (IEEE-CIS Fraud Detection dataset).
- [x] Feature availability, derivable features, and unavailable fields documented.
- [x] Advanced dual-engine architecture proposed.
- [x] Migration and leakage protection strategy defined.

**STOP GATE A REACHED — WAITING FOR USER APPROVAL BEFORE PROCEEDING.**
