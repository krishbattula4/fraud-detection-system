# AI Financial Fraud Detection & Risk Intelligence System

A multi-model financial fraud detection and risk intelligence platform featuring dual evaluation architectures: a protected baseline model trained on the ULB European Cardholder dataset and an advanced multi-entity risk engine trained on authentic IEEE-CIS payment processing data (Vesta Corp benchmark).

---

## 1. System Pipeline Architecture

```
                               TRANSACTION INGESTION
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      Baseline ULB Endpoint                       Advanced IEEE-CIS Endpoint
        (POST /predict)                             (POST /predict/advanced)
                   │                                           │
         Schema Validation                           Schema Validation
        (TransactionInput)                       (AdvancedTransactionInput)
                   │                                           │
        DataPreprocessor                            Stateful Entity Engine
    (RobustScaler: Time/Amount)                  (Sliding-Window Velocity & Novelty)
                   │                                           │
   ┌───────────────┼───────────────┐           ┌───────────────┼───────────────┐
   ▼               ▼               ▼           ▼               ▼               ▼
XGBoost        Isolation         Local     Calibrated      Isolation       Behavioral
(Calibrated)   Forest           Outlier     XGBoost          Forest         Velocity
   │               │            Factor         │               │               │
   └───────────────┼───────────────┘           └───────────────┼───────────────┘
                   ▼                                           ▼
          Hybrid Risk Engine                     Advanced Hybrid Risk Engine
       (Weights: 60 / 25 / 15)                     (Weights: 60 / 25 / 15)
                   │                                           │
         0.0–100.0 Risk Score                       0.0–100.0 Risk Score
                   │                                           │
         Risk Tier & Decision                       Risk Tier & Decision
        (LOW/MED/HIGH/CRITICAL)                    (LOW/MED/HIGH/CRITICAL)
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                             SQLAlchemy ORM Persistence
                                         │
                             Streamlit Risk Dashboard
```

---

## 2. Dual Subsystem Architecture

| Subsystem | Baseline ULB Subsystem | Authentic IEEE-CIS Advanced Subsystem |
| :--- | :--- | :--- |
| **Model Version** | `1.0.0` | `2.0.0-authentic-ieee` |
| **API Endpoint** | `POST /predict` | `POST /predict/advanced` |
| **Artifact Directory** | `./models/artifacts/` | `./models/advanced_artifacts/` |
| **Dataset Source** | ULB European Cardholders (`creditcard.csv`) | Authentic Kaggle IEEE-CIS Fraud Detection (Vesta Corp) |
| **Input Schema** | 31 features (`Time`, `V1`–`V28`, `Amount`) | 12 features + Entity Context (`card_id`, domains, device) |
| **Feature Processing** | `RobustScaler` on `Time` and `Amount` | Stateful sliding-window velocity & device/domain novelty |
| **Supervised Model** | Calibrated XGBoost (`scale_pos_weight=518.2x`) | Calibrated Advanced XGBoost |
| **Unsupervised Models**| Isolation Forest + Local Outlier Factor | Advanced Isolation Forest + Behavioral Velocity Engine |

---

## 3. Hybrid Risk Engine & Decision Scoring

The system calculates a calibrated decision-ranking score bounded between **0.0 and 100.0**.

> [!IMPORTANT]
> The **0–100 Risk Score** is an internal decision-support risk score, NOT a literal fraud probability.

### Current Implemented Component Weights:
- **Calibrated XGBoost (Supervised Fraud Prob)**: `60%` ($0.60$)
- **Isolation Forest (Unsupervised Anomaly Score)**: `25%` ($0.25$)
- **Behavioral Velocity / LOF Signal**: `15%` ($0.15$)

*Note: Component weights reflect the current implemented engine configuration and are not claimed to be globally optimal.*

### Decision Action & Risk Band Mapping:

| Score Range | Risk Level Tier | Automated Decision Action | Operational Workflow |
| :--- | :--- | :--- | :--- |
| `0.0 – 39.99` | `LOW` | `APPROVE` | Transaction approved automatically |
| `40.0 – 69.99` | `MEDIUM` | `FLAG_FOR_REVIEW` | Queued for analyst inspection |
| `70.0 – 89.99` | `HIGH` | `FLAG_FOR_REVIEW` | High-priority risk alert created |
| `90.0 – 100.0` | `CRITICAL` | `DECLINE` | Transaction declined automatically |

---

## 4. Empirical Model Evaluation Benchmark

Evaluated on untouched held-out test split partitions:

### Authentic IEEE-CIS Advanced System (v2.0.0-authentic-ieee):
- **Advanced XGBoost (Calibrated)**: PR-AUC `0.3346` | ROC-AUC `0.8050` | Precision `73.25%` | Recall `25.40%` | F1 `0.3772`
- **Advanced Isolation Forest**: PR-AUC `0.0591` | ROC-AUC `0.6424` | Precision `5.95%` | Recall `9.31%` | F1 `0.0726`

### ULB Baseline System (v1.0.0):
- **Random Forest**: PR-AUC `0.7716` | ROC-AUC `0.9608` | Precision `95.12%` | Recall `75.00%` | F1 `0.8387`
- **XGBoost (Calibrated)**: PR-AUC `0.7508` | ROC-AUC `0.9675` | Precision `85.71%` | Recall `69.23%` | F1 `0.7660`
- **Hybrid Risk Engine**: PR-AUC `0.6648` | ROC-AUC `0.9399` | Precision `85.71%` | Recall `69.23%` | F1 `0.7660`

---

## 5. Repository Structure

```
├── creditcard.csv                 # ULB Kaggle baseline dataset
├── models/
│   ├── artifacts/                 # Baseline v1.0.0 joblib artifacts
│   └── advanced_artifacts/        # Advanced v2.0.0 authentic IEEE-CIS joblib artifacts
├── src/
│   ├── api/                       # FastAPI REST API endpoints
│   ├── core/                      # Configuration settings and structured logging
│   ├── data/                      # Leakage-safe preprocessors and entity velocity engines
│   ├── db/                        # SQLAlchemy ORM models and repository pattern
│   ├── dashboard/                 # Streamlit multi-page risk intelligence center
│   ├── engine/                    # Baseline & Advanced Hybrid Risk Engines
│   ├── explainability/            # SHAP feature attribution engines
│   ├── ml/                        # Supervised, anomaly, and calibration wrappers
│   └── schemas/                   # Pydantic validation request/response schemas
├── tests/                         # Unit and integration test suite (84 tests)
└── pyproject.toml                 # Dependencies and pytest configuration
```

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+ runtime environment

### Setup Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute full automated test suite
python -m pytest tests/ -v

# 3. Start FastAPI REST Backend (Port 8000)
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# 4. Launch Streamlit Risk Intelligence Dashboard (Port 8501)
streamlit run src/dashboard/app.py
```

---

## 7. System Disclosures & Prototype Limitations

1. **Prototype Scope**: Designed for demonstration and technical evaluation; not connected to live banking authorization networks or payment gateways.
2. **Security Safeguards**: Implements Pydantic validation and input hardening; not intended as production-grade network security.
3. **No Fraud Detection Guarantee**: Risk scores are statistical estimations; 100% fraud detection accuracy is not guaranteed.
4. **Anonymized Features**: Baseline features `V1`–`V28` are anonymized principal components derived via PCA to preserve cardholder privacy.
