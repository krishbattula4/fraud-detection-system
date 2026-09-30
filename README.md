FraudLens — Fraud Detection & Risk Intelligence

Detect. Analyze. Protect.

FraudLens is an end-to-end financial fraud detection and risk intelligence platform that combines machine learning, anomaly detection, behavioral analysis, temporal entity intelligence, explainability, REST APIs, alert workflows, and a deployed interactive dashboard.

The system provides two complementary evaluation architectures:

Baseline ULB Fraud Detection — trained around the ULB European Credit Card Fraud benchmark.
Advanced Multi-Entity Risk Intelligence — built around authentic IEEE-CIS payment-processing data with behavioral and entity-level signals.

The goal is not simply to classify a transaction as fraudulent or legitimate, but to transform multiple risk signals into an interpretable 0–100 decision-support risk score, a risk tier, and an operational action.

🚀 Live Application
FraudLens Dashboard

Live Application:
https://fraud-detection-system-ucb3zmzfppvaqnwe6cu5gn.streamlit.app/

FastAPI Backend

API:
https://fraud-detection-system-z5g1.onrender.com/

Health Check:
https://fraud-detection-system-z5g1.onrender.com/health

API Documentation:
https://fraud-detection-system-z5g1.onrender.com/docs

The frontend is deployed on Streamlit Community Cloud and communicates with the FastAPI backend deployed on Render.

✨ What Makes FraudLens Different?

FraudLens goes beyond a conventional:

Transaction → ML Model → Fraud / Not Fraud

Instead, the system follows:

Transaction
↓
Feature Processing
↓
ML Detection + Entity / Behavioral Intelligence
↓
Hybrid Risk Engine
↓
0–100 Risk Score
↓
Risk Tier + Decision
↓
Alert / Investigation

Key capabilities
Multi-model fraud detection
Supervised + unsupervised learning
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
Automated test coverage
🏗️ System Architecture

TRANSACTION INGESTION

                ┌──────────────────────────────┐
                │      Transaction Input       │
                └──────────────┬───────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    BASELINE ULB PIPELINE             ADVANCED IEEE-CIS PIPELINE
              │                                 │
      POST /predict                    POST /predict/advanced
              │                                 │
              ▼                                 ▼
    Schema Validation                  Schema Validation
              │                                 │
              ▼                                 ▼
    Data Preprocessing                Stateful Entity Engine
    RobustScaler                      Velocity & Novelty
              │                                 │
    ┌─────────┼─────────┐              ┌────────┼─────────┐
    ▼         ▼         ▼              ▼        ▼         ▼
 XGBoost      IF        LOF          XGBoost     IF    Behavioral
    │         │         │              │        │      Velocity
    └─────────┼─────────┘              └────────┼────────┘
              │                                 │
              ▼                                 ▼
      Hybrid Risk Engine              Advanced Risk Engine
              │                                 │
              └──────────────┬──────────────────┘
                             ▼
                      0–100 Risk Score
                             │
                             ▼
                  Risk Tier + Decision
                             │
                             ▼
                     Alert / Review
                             │
                             ▼
                  SQLAlchemy Persistence
                             │
                             ▼
                   Streamlit Risk Center
🔀 Dual Risk Intelligence Architecture
Component	Baseline ULB	Advanced IEEE-CIS
Model Version	1.0.0	2.0.0-authentic-ieee
API	POST /predict	POST /predict/advanced
Dataset	ULB European Cardholder benchmark	Authentic IEEE-CIS / Vesta benchmark
Input	Time, V1–V28, Amount	Transaction + entity/context features
Preprocessing	RobustScaler	Stateful feature engineering
Supervised Model	Calibrated XGBoost	Calibrated Advanced XGBoost
Anomaly Model	Isolation Forest	Advanced Isolation Forest
Behavioral Signal	LOF / anomaly signal	Entity velocity & behavioral signals
Entity Intelligence	—	Customer/device/domain context
Temporal Intelligence	—	Prior-transaction state
Output	Risk score + decision	Risk score + decision
🧠 Advanced Multi-Entity Risk Intelligence

The advanced subsystem introduces contextual information that is not available from an isolated transaction alone.

Behavioral features

The system evaluates signals such as:

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
Temporal entity intelligence

Entity features are calculated from transaction history while enforcing:

previous transaction time < current transaction time

This prevents future transaction information from being used as historical evidence for the transaction being evaluated.

⚖️ Hybrid Risk Engine

FraudLens converts multiple signals into a bounded:

0.0 ─────────────────────────────── 100.0

risk score.

Important: The 0–100 value is an internal decision-support risk score. It is not a literal probability of fraud.

Implemented component weighting
Component	Weight
Calibrated XGBoost	60%
Isolation Forest	25%
Behavioral / Velocity Signal	15%

These weights represent the current implemented engine configuration and are not claimed to be globally optimal.

🚦 Risk Levels & Actions
Risk Score	Risk Level	Decision	Operational Meaning
0–39.99	LOW	APPROVE	Normal transaction
40–69.99	MEDIUM	FLAG_FOR_REVIEW	Analyst review
70–89.99	HIGH	FLAG_FOR_REVIEW	High-priority investigation
90–100	CRITICAL	DECLINE	High-risk transaction
🤖 Machine Learning
Baseline Models

FraudLens includes:

Logistic Regression
Random Forest
Calibrated XGBoost
Isolation Forest
Local Outlier Factor
Baseline evaluation

Evaluated on an untouched chronological test partition.

Model	PR-AUC	ROC-AUC	Precision	Recall	F1
Random Forest	0.7716	0.9608	95.12%	75.00%	0.8387
XGBoost	0.7508	0.9675	85.71%	69.23%	0.7660
Hybrid Risk Engine	0.6648	0.9399	85.71%	69.23%	0.7660
🔬 Advanced IEEE-CIS System

The advanced subsystem is based on authentic IEEE-CIS payment-processing data associated with the Vesta Corp benchmark.

Dataset

Transactions: 590,540
Fraudulent: 20,663
Fraud Rate: 3.4990%

Chronological split

Training: 413,378 transactions / 14,538 fraud
Validation: 88,581 transactions / 3,042 fraud
Test: 88,581 transactions / 3,083 fraud

Advanced evaluation
Model	PR-AUC	ROC-AUC	Precision	Recall	F1
Advanced XGBoost	0.3346	0.8050	73.25%	25.40%	0.3772
Advanced Isolation Forest	0.0591	0.6424	5.95%	9.31%	0.0726

All reported metrics are based on the held-out evaluation partitions used by the project.

🔍 Explainability

FraudLens provides risk explanations alongside prediction results.

The system can surface information related to:

model contribution
anomaly signals
behavioral risk signals
transaction characteristics
rule-triggered conditions

This allows the application to provide more context than a binary fraud label.

🚨 Alert & Investigation Workflow

FraudLens connects model output to an operational workflow:

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

The Streamlit application provides dedicated workspaces for:

Overview

System-level risk and activity information.

Investigate

Transaction and risk investigation workflows.

Risk Check

Interactive transaction risk evaluation.

Available evaluation modes:

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

Instead, the deployed application contains a compact benchmark sample:

data/benchmark_samples/baseline_benchmark_samples.json

The packaged sample contains:

75 authentic records
├── 50 legitimate
└── 25 fraud

This allows the deployed Benchmark Dataset Mode to function without uploading the full raw dataset to GitHub.

The historical Class value is used only as benchmark context and is not included in the live prediction payload.

🧪 Testing

The project includes unit and integration tests covering core application behavior.

Current verified test result:

84 passed
19 warnings
0 failures

The warnings include serialized-model version compatibility warnings associated with previously trained scikit-learn and XGBoost artifacts.

🧰 Technology Stack
Frontend
Streamlit
Python
Custom CSS/UI components
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
Development & Version Control
Python 3.11
uv
WSL2 Ubuntu
Git
GitHub
Git LFS
📁 Repository Structure

fraud-detection-system/

├── data/
│ └── benchmark_samples/
│ └── baseline_benchmark_samples.json
│
├── models/
│ ├── artifacts/
│ │ └── baseline model artifacts
│ │
│ └── advanced_artifacts/
│ └── advanced model artifacts
│
├── src/
│ ├── api/
│ │ └── FastAPI routes and application
│ │
│ ├── core/
│ │ └── configuration and logging
│ │
│ ├── data/
│ │ └── preprocessing and entity intelligence
│ │
│ ├── db/
│ │ └── SQLAlchemy models and repositories
│ │
│ ├── dashboard/
│ │ └── Streamlit application
│ │
│ ├── engine/
│ │ └── baseline and advanced risk engines
│ │
│ ├── explainability/
│ │ └── explanation and attribution logic
│ │
│ ├── ml/
│ │ └── model wrappers and inference
│ │
│ └── schemas/
│ └── Pydantic request/response schemas
│
├── tests/
│ ├── unit/
│ └── integration/
│
├── notebooks/
│ └── fraud detection research/training notebook
│
├── pyproject.toml
├── .gitignore
├── .gitattributes
└── README.md

The full raw benchmark datasets are intentionally excluded from version control.

⚙️ Local Installation
Prerequisites
Python 3.10+
Git
Git LFS
Install dependencies

pip install -r requirements.txt

Or, using the project's uv environment:

uv sync

🧪 Run Tests

python -m pytest tests/ -v

Or:

uv run python -m pytest tests/ -v

🚀 Run the Backend

python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

The API will be available at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

🖥️ Run the Dashboard

In a separate terminal:

streamlit run src/dashboard/app.py

The dashboard will normally be available at:

http://localhost:8501

🔌 API Endpoints
Method	Endpoint	Purpose
GET	/health	Backend health
GET	/metrics	System/model metrics
POST	/predict	Baseline transaction risk
POST	/predict/advanced	Advanced multi-entity risk
GET	/alerts	Retrieve alerts
POST	/alerts/{id}/review	Review an alert
GET	/transactions	Retrieve transactions
🔐 Security & Data Handling

FraudLens follows several safeguards:

Pydantic request validation
Input validation and hardening
Raw datasets excluded from Git
Database files excluded from Git
Environment secrets excluded from Git
Git LFS for large model artifacts
Benchmark deployment sample separated from the raw dataset
Historical fraud labels are not sent as live prediction inputs

The baseline V1–V28 features are anonymized principal components provided by the benchmark dataset.

⚠️ Limitations
FraudLens is an analytical and decision-support system and is not connected to live banking authorization networks or payment gateways.
Risk scores are statistical decision-support outputs and do not guarantee fraud detection.
The baseline dataset uses anonymized PCA-derived features (V1–V28).
The current deployment uses SQLite persistence. Cloud instance storage may not provide durable long-term persistence across infrastructure recreation.
Model serialization warnings may occur when trained artifacts are loaded under different library versions. The current automated test suite completes successfully.
Production deployment in a regulated financial environment would require additional security, monitoring, governance, compliance, model validation, and persistent infrastructure.
📌 Project Status

Backend ✅ Complete
ML Pipelines ✅ Complete
Risk Engine ✅ Complete
Entity Intelligence ✅ Complete
Alert Workflow ✅ Complete
Streamlit UI ✅ Complete
Automated Tests ✅ 84 Passing
Cloud Deployment ✅ Deployed
Git/Git LFS ✅ Configured
Benchmark Mode ✅ Deployment Ready

👨‍💻 Project

FraudLens — Fraud Detection & Risk Intelligence

Tagline:

Detect. Analyze. Protect.
