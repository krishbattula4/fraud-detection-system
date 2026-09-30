# AGENTS.md — AI Agent Guidelines & Operating Rules

## Project Purpose
This repository implements the **AI Financial Fraud Detection & Risk Intelligence System**.
All future modifications must adhere strictly to software engineering best practices, data science rigor, and the project architectural boundaries established in Phase 1.

---

## MANDATORY AGENT RULES

### 1. Data Integrity & Realism
- **NO FAKE DATA IN CORE PIPELINES:** Do not invent fake data, hard-coded placeholder scores, or simulated response logic in core models or backend services.
- **NO DESTRUCTIVE FALLBACKS:** Never swallow exceptions with dummy fallback values (e.g. returning 0-byte arrays or hardcoded zero scores) when APIs fail.

### 2. Strict ML Architecture & Data Leakage Prevention
- **SEPARATE TRAINING FROM INFERENCE:** Model training pipelines must be strictly decoupled from production inference services. Training code MUST NOT execute when the API starts.
- **NO TEST LEAKAGE:** Scalers, imputers, anomaly detectors, and feature transformers MUST be fitted strictly on training data splits. Never fit transformers on validation or test sets.
- **CHRONOLOGICAL EVALUATION:** Respect temporal transaction ordering (`Time` feature) during dataset splitting.

### 3. Module Boundaries & Architecture
- **API ROUTERS:** Handle request validation, parameter parsing, and delegating to services. Never include model training logic inside API route handlers.
- **UI / DASHBOARD:** Consume application service layer or REST API endpoints. Streamlit components must NOT directly query database internals or manipulate ML models directly.
- **PERSISTENCE:** Database interactions must pass through SQLAlchemy repositories and session dependencies.

### 4. Security & Environment
- **NO SECRETS IN SOURCE:** Never hard-code secrets, passwords, or API keys in source code. All configuration must load via environment variables (`pydantic-settings`).
- **NEVER COMMIT REPO SECRET FILES:** Ensure `.env` is ignored by Git.

### 5. Verification & Testing
- **VERIFY BEFORE COMPLETION:** Run unit tests (`pytest`), format checks, and import checks before completing any task.
- **NO PLACEHOLDER TESTS:** Write meaningful assertions verifying contract boundaries, schemas, and error cases.

---

## Technical Stack & Commands
- **Python Runtime:** Python 3.10+
- **Test Runner:** `python -m pytest tests/ -v`
- **Linting / Import Verification:** `python -c "import src"`
