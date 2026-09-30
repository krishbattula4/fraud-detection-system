FraudLens — Fraud Detection & Risk Intelligence

Detect. Analyze. Protect.

FraudLens is an end-to-end financial fraud detection and risk intelligence platform that combines machine learning, anomaly detection, behavioral analysis, temporal entity intelligence, explainability, REST APIs, alert workflows, and an interactive cloud dashboard.

The system provides two complementary architectures:

Baseline ULB Fraud Detection — based on the ULB European Credit Card Fraud benchmark.

Advanced Multi-Entity Risk Intelligence — based on authentic IEEE-CIS / Vesta payment-processing data with behavioral and entity-level signals.

The objective is to move beyond a simple fraud/not-fraud classification and produce an interpretable 0–100 decision-support risk score, risk tier, and operational action.

🚀 Live Application

FraudLens Dashboard

Frontend:
https://fraud-detection-system-ucb3zmzfppvaqnwe6cu5gn.streamlit.app/

FastAPI Backend

API Documentation:
https://fraud-detection-system-z5g1.onrender.com/docs

Health Check:
https://fraud-detection-system-z5g1.onrender.com/health

Metrics:
https://fraud-detection-system-z5g1.onrender.com/metrics

The frontend is deployed on Streamlit Community Cloud and communicates with the FastAPI backend deployed on Render.

✨ Key Capabilities

Multi-model fraud detection

Supervised and unsupervised learning

Behavioral velocity analysis

Entity-level transaction intelligence

Temporal leakage protection

Device and email-domain novelty detection

Hybrid risk scoring

Risk-based operational decisions

Explainable risk outputs

Alert and investigation workflow

REST API integration

Cloud deployment

Automated testing

Deployment-safe benchmark evaluation

🏗️ System Architecture

Transaction Input
│
▼
┌───────────────────────────────┐
│ Validation & Preprocessing     │
└───────────────┬───────────────┘
│
┌────────┴─────────┐
│                  │
▼                  ▼
Baseline ULB          Advanced IEEE-CIS
Pipeline              Multi-Entity Pipeline
│                  │
▼                  ▼
XGBoost / IF / LOF    XGBoost / IF / Behavioral
│                  │
└────────┬─────────┘
▼
Hybrid Risk Engine
│
▼
0–100 Risk Score
│
▼
Risk Tier + Decision
│
▼
Alert / Investigation
│
▼
Persistence Layer
│
▼
FraudLens Dashboard

🔀 Dual Risk Intelligence Architecture

Component

Baseline ULB

Advanced IEEE-CIS

Version

1.0.0

2.0.0-authentic-ieee

API

/predict

/predict/advanced

Dataset

ULB European Credit Card Fraud

IEEE-CIS / Vesta

Input

Time, V1–V28, Amount

Transaction + entity/context features

Preprocessing

RobustScaler

Stateful feature engineering

Supervised Model

Calibrated XGBoost

Calibrated Advanced XGBoost

Anomaly Model

Isolation Forest

Advanced Isolation Forest

Behavioral Signal

Anomaly signal

Entity velocity and behavioral signals

Entity Intelligence

—

Customer/device/domain context

Temporal Intelligence

—

Prior-transaction state

Output

Risk score + decision

Risk score + decision

🧠 Advanced Multi-Entity Risk Intelligence

The advanced subsystem evaluates contextual information surrounding a transaction rather than relying only on the transaction in isolation.

Behavioral Features

The system evaluates signals including:

cust_tx_count_1h

cust_tx_count_24h

cust_amt_sum_24h

amt_to_cust_avg_ratio

is_new_device_for_cust

is_new_email_domain

hour_of_day

day_of_week

c1

d1

Temporal Entity Intelligence

Entity features are calculated using transaction history while enforcing:

previous_transaction_time < current_transaction_time

This prevents future transaction information from being used as historical evidence for the transaction being evaluated.

⚖️ Hybrid Risk Engine

FraudLens combines multiple signals into a bounded 0–100 decision-support risk score.

The current advanced configuration uses:

Component

Weight

Calibrated XGBoost

60%

Isolation Forest

25%

Behavioral / Velocity Signal

15%

The 0–100 score is a decision-support score and should not be interpreted as a literal probability of fraud.

🚦 Risk Levels & Actions

Risk Score

Risk Level

Decision

0–39.99

LOW

APPROVE

40–69.99

MEDIUM

FLAG_FOR_REVIEW

70–89.99

HIGH

FLAG_FOR_REVIEW

90–100

CRITICAL

DECLINE

🤖 Machine Learning

Baseline Models

FraudLens includes:

Logistic Regression

Random Forest

Calibrated XGBoost

Isolation Forest

Local Outlier Factor

Baseline Evaluation

The baseline models were evaluated using a chronological train/validation/test design.

Model

PR-AUC

ROC-AUC

Precision

Recall

F1

Random Forest

0.7716

0.9608

95.12%

75.00%

0.8387

XGBoost

0.7508

0.9675

85.71%

69.23%

0.7660

Hybrid Risk Engine

0.6648

0.9399

85.71%

69.23%

0.7660

🔬 Advanced IEEE-CIS System

The advanced subsystem uses authentic IEEE-CIS / Vesta payment-processing data.

Dataset

Transactions: 590,540

Fraudulent transactions: 20,663

Fraud rate: 3.4990%

Chronological Split

Training: 413,378 transactions / 14,538 fraud

Validation: 88,581 transactions / 3,042 fraud

Test: 88,581 transactions / 3,083 fraud

Advanced Evaluation

Model

PR-AUC

ROC-AUC

Precision

Recall

F1

Advanced XGBoost

0.3346

0.8050

73.25%

25.40%

0.3772

Advanced Isolation Forest

0.0591

0.6424

5.95%

9.31%

0.0726

🔍 Explainability

FraudLens provides contextual information alongside risk results, including:

Model contribution

Anomaly signals

Behavioral risk signals

Transaction characteristics

Triggered risk conditions

This provides more context than a simple binary fraud label.

🚨 Alert & Investigation Workflow

The system connects model output to an operational investigation flow:

Transaction
↓
Risk Evaluation
↓
Risk Score
↓
Risk Level
↓
Decision
↓
Alert
↓
Investigation / Review

The backend supports alert retrieval and review operations through the API.

🖥️ Application Workspaces

The FraudLens dashboard provides:

Overview

System-level activity, risk distribution, and operational metrics.

Investigate

Transaction and alert investigation workflows.

Risk Check

Interactive transaction risk evaluation.

Available modes:

Benchmark Dataset Mode

Custom PCA Vector Mode

Advanced Multi-Entity Risk Intelligence

Transactions

Transaction-level information and risk results.

Intelligence

Advanced behavioral and risk intelligence.

Status

Application and backend health information.

📦 Deployment-Safe Benchmark Dataset

The complete raw ULB dataset is intentionally excluded from the Git repository.

Instead, the deployed application contains:

data/benchmark_samples/baseline_benchmark_samples.json

The packaged benchmark sample contains:

75 authentic records

50 legitimate

25 fraud

This allows Benchmark Dataset Mode to operate in the deployed application without uploading the full raw dataset to GitHub.

The historical fraud label is used as benchmark context and is not included in the live prediction payload.

🧪 Testing

The project includes unit and integration tests covering the core application.

Current verified result:

84 passed
19 warnings
0 failures

The warnings are associated with serialized scikit-learn and XGBoost artifacts and library-version compatibility.

🧰 Technology Stack

Frontend

Streamlit

Python

Custom UI/CSS

Backend

FastAPI

Uvicorn

Pydantic

Machine Learning

Scikit-learn

XGBoost

Isolation Forest

Local Outlier Factor

SHAP-based explainability

Data Processing

Pandas

NumPy

Database

SQLite

SQLAlchemy

Deployment

Streamlit Community Cloud

Render

Development

Python 3.11

uv

WSL2 Ubuntu

Git

GitHub

Git LFS

📁 Repository Structure

fraud-detection-system/

├── data/
│   └── benchmark_samples/
│       └── baseline_benchmark_samples.json
│
├── models/
│   ├── artifacts/
│   │   └── baseline model artifacts
│   └── advanced_artifacts/
│       └── advanced model artifacts
│
├── src/
│   ├── api/
│   │   └── FastAPI application and routes
│   ├── core/
│   │   └── configuration and logging
│   ├── data/
│   │   └── preprocessing and entity intelligence
│   ├── db/
│   │   └── SQLAlchemy models and repositories
│   ├── dashboard/
│   │   └── Streamlit application
│   ├── engine/
│   │   └── baseline and advanced risk engines
│   ├── explainability/
│   │   └── explanation and attribution logic
│   ├── ml/
│   │   └── model wrappers and inference
│   └── schemas/
│       └── Pydantic request/response schemas
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── notebooks/
│   └── fraud detection research/training notebook
│
├── pyproject.toml
├── .gitignore
├── .gitattributes
└── README.md

Raw benchmark datasets are intentionally excluded from version control.

⚙️ Local Installation

Prerequisites

Python 3.10+

Git

Git LFS

Install Dependencies

Using pip:

pip install -r requirements.txt

Or using uv:

uv sync

🧪 Run Tests

python -m pytest tests/ -v

Or:

uv run python -m pytest tests/ -v

🚀 Run the Backend

python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

🖥️ Run the Dashboard

streamlit run src/dashboard/app.py

Dashboard:

http://localhost:8501

🔌 API Endpoints

Method

Endpoint

Purpose

GET

/health

Backend health

GET

/metrics

System and evaluation metrics

POST

/predict

Baseline transaction risk

POST

/predict/advanced

Advanced multi-entity risk

GET

/alerts

Retrieve alerts

POST

/alerts/{id}/review

Review an alert

GET

/transactions

Retrieve transactions

Interactive API documentation is available at:

https://fraud-detection-system-z5g1.onrender.com/docs

🔐 Security & Data Handling

FraudLens includes:

Pydantic request validation

Input validation

Raw datasets excluded from Git

Database files excluded from Git

Environment secrets excluded from Git

Git LFS for large model artifacts

Deployment-safe benchmark sample

Separation of benchmark labels from live prediction payloads

The baseline V1–V28 features are anonymized principal components provided by the benchmark dataset.

⚠️ Limitations

FraudLens is an analytical and decision-support system and is not connected to live banking authorization networks or payment gateways.

Risk scores are statistical decision-support outputs and do not guarantee fraud detection.

The baseline dataset uses anonymized PCA-derived features.

The current deployment uses SQLite persistence. Cloud instance storage may not provide durable long-term persistence across infrastructure recreation.

Model serialization warnings can occur when trained artifacts are loaded under different library versions.

A production financial deployment would require additional security, monitoring, governance, compliance, model validation, and durable infrastructure.

📌 Project Status

Backend              ✅ Complete
ML Pipelines         ✅ Complete
Risk Engine          ✅ Complete
Entity Intelligence  ✅ Complete
Alert Workflow       ✅ Complete
Streamlit UI         ✅ Complete
Automated Tests      ✅ 84 Passing
Cloud Deployment     ✅ Deployed
Git / Git LFS         ✅ Configured
Benchmark Mode       ✅ Deployment Ready

🌐 Deployment

Live Frontend:
https://fraud-detection-system-ucb3zmzfppvaqnwe6cu5gn.streamlit.app/

FastAPI Documentation:
https://fraud-detection-system-z5g1.onrender.com/docs

Backend Health:
https://fraud-detection-system-z5g1.onrender.com/health

👨‍💻 Project

FraudLens — Fraud Detection & Risk Intelligence

Detect. Analyze. Protect.
