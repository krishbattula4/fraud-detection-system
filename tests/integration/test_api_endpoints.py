"""Integration test suite for Phase 4 FastAPI REST API endpoints."""
import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import create_app
from src.db.session import Base, get_db_session
from src.api.routes.predict import get_prediction_service
from src.engine.prediction_service import PredictionService
from src.data.preprocessor import DataPreprocessor, FEATURE_COLUMNS
from src.ml.supervised import XGBoostFraudModel
from src.ml.anomaly import IsolationForestAnomalyDetector, LOFAnomalyDetector
from src.ml.calibration import ProbabilityCalibrator, MinMaxScoreNormalizer
from src.explainability.explainer import SHAPTransactionExplainer
from src.ml.registry import ModelRegistry
from src.schemas.prediction import RiskLevel
from src.schemas.alert import AlertStatus


@pytest.fixture
def test_db_engine():
    """Shared in-memory SQLite database engine fixture using StaticPool."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def mock_prediction_service():
    """Synthetic trained PredictionService instance fixture."""
    np.random.seed(42)
    X_raw = np.random.normal(loc=0.0, scale=1.0, size=(100, 30))
    y_raw = np.zeros(100, dtype=int)
    y_raw[:10] = 1

    df_train = pd.DataFrame(X_raw, columns=FEATURE_COLUMNS)
    preprocessor = DataPreprocessor().fit(df_train)
    X_trans = preprocessor.transform(df_train)

    xgb = XGBoostFraudModel(n_estimators=10).fit(X_trans, y_raw)
    iforest = IsolationForestAnomalyDetector(n_estimators=10).fit(X_trans)
    lof = LOFAnomalyDetector(n_neighbors=5).fit(X_trans)

    uncal_probs = xgb.predict_proba(X_trans)[:, 1]
    calibrator = ProbabilityCalibrator().fit(uncal_probs, y_raw)

    if_scores = iforest.predict_anomaly_score(X_trans)
    norm_if = MinMaxScoreNormalizer().fit(if_scores)

    lof_scores = lof.predict_anomaly_score(X_trans)
    norm_lof = MinMaxScoreNormalizer().fit(lof_scores)

    explainer = SHAPTransactionExplainer().fit_explainer(xgb, background_data=X_trans[:10])

    return PredictionService(
        preprocessor=preprocessor,
        xgb_model=xgb,
        iforest_model=iforest,
        lof_model=lof,
        calibrator=calibrator,
        iforest_normalizer=norm_if,
        lof_normalizer=norm_lof,
        explainer=explainer,
        alert_levels={RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL},
    )


@pytest.fixture
def valid_transaction_payload():
    """Valid transaction JSON payload matching ULB Credit Card dataset schema."""
    payload = {
        "transaction_id": "TX-API-TEST-001",
        "time": 123.45,
        "amount": 499.99,
    }
    for i in range(1, 29):
        payload[f"v{i}"] = 0.05 * i
    return {"transaction": payload}


@pytest.fixture
def api_client(test_db_engine, mock_prediction_service):
    """FastAPI TestClient with DB session and PredictionService dependency overrides."""
    app = create_app()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)

    def _override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _override_get_service():
        return mock_prediction_service

    app.dependency_overrides[get_db_session] = _override_get_db
    app.dependency_overrides[get_prediction_service] = _override_get_service

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# =====================================================================
# API Endpoints Integration Tests (16 - 24)
# =====================================================================

def test_predict_success_path(api_client, valid_transaction_payload):
    """16. Test POST /predict success path returning valid RiskPredictionResponse and persisting entities."""
    response = api_client.post("/predict", json=valid_transaction_payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction_id" in data
    assert data["transaction_id"] == "TX-API-TEST-001"
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data["decision"] in ["APPROVE", "FLAG_FOR_REVIEW", "DECLINE"]
    assert "xgboost_calibrated_prob" in data["model_components"]
    assert "isolation_forest_anomaly_score" in data["model_components"]
    assert "lof_anomaly_score" in data["model_components"]
    assert isinstance(data["explanations"], list)
    assert len(data["explanations"]) == 5


def test_invalid_transaction_request(api_client):
    """17. Test POST /predict with invalid transaction request (missing required fields / negative amount)."""
    # Missing V1..V28
    bad_payload_1 = {"transaction": {"time": 10.0, "amount": 100.0}}
    res1 = api_client.post("/predict", json=bad_payload_1)
    assert res1.status_code == 422

    # Negative amount
    bad_payload_2 = {"transaction": {"time": 10.0, "amount": -50.0, "v1": 0.0}}
    res2 = api_client.post("/predict", json=bad_payload_2)
    assert res2.status_code == 422


def test_missing_artifacts_returns_503(test_db_engine, valid_transaction_payload):
    """18. Test POST /predict returns 503 Service Unavailable when model artifacts are missing."""
    app = create_app()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)

    def _override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Uninitialized PredictionService (no injected models & empty registry)
    empty_registry = ModelRegistry(model_dir="./models/empty_test_artifacts")
    uninitialized_service = PredictionService(registry=empty_registry)

    app.dependency_overrides[get_db_session] = _override_get_db
    app.dependency_overrides[get_prediction_service] = lambda: uninitialized_service

    client = TestClient(app)
    response = client.post("/predict", json=valid_transaction_payload)
    
    assert response.status_code == 503
    assert "Model service unavailable" in response.json()["detail"]


def test_get_alerts(api_client, valid_transaction_payload):
    """19. Test GET /alerts retrieving persisted alerts with status filtering and pagination."""
    # Execute a prediction that generates an alert
    api_client.post("/predict", json=valid_transaction_payload)

    response = api_client.get("/alerts")
    assert response.status_code == 200
    alerts_list = response.json()
    assert isinstance(alerts_list, list)
    assert len(alerts_list) >= 1

    filtered_response = api_client.get("/alerts?status=OPEN&limit=10&offset=0")
    assert filtered_response.status_code == 200
    assert isinstance(filtered_response.json(), list)


def test_get_transaction_by_id(api_client, valid_transaction_payload):
    """20. Test GET /transactions/{transaction_id} retrieving stored transaction and prediction info."""
    # Post a prediction first
    pred_res = api_client.post("/predict", json=valid_transaction_payload)
    assert pred_res.status_code == 200

    # Query by client transaction ID
    tx_res = api_client.get("/transactions/TX-API-TEST-001")
    assert tx_res.status_code == 200
    tx_data = tx_res.json()
    assert tx_data["transaction_id"] == "TX-API-TEST-001"
    assert tx_data["amount"] == 499.99
    assert tx_data["prediction"] is not None
    assert "risk_score" in tx_data["prediction"]


def test_post_alert_review(api_client, valid_transaction_payload):
    """21. Test POST /alerts/{alert_id}/review submitting analyst investigation action."""
    # Run prediction to trigger alert
    api_client.post("/predict", json=valid_transaction_payload)

    # Fetch alert ID
    alerts_res = api_client.get("/alerts")
    alerts = alerts_res.json()
    assert len(alerts) >= 1

    target_alert_id = alerts[0]["alert_id"]

    review_payload = {
        "status": "CONFIRMED_FRAUD",
        "notes": "Verified fraudulent activity with cardholder bank.",
        "analyst_id": "ANALYST-SEC-01",
    }

    review_res = api_client.post(f"/alerts/{target_alert_id}/review", json=review_payload)
    assert review_res.status_code == 200
    updated_data = review_res.json()
    assert updated_data["alert_id"] == target_alert_id
    assert updated_data["status"] == "CONFIRMED_FRAUD"
    assert "Verified fraudulent" in updated_data["reviewer_notes"]


def test_unknown_alert_id_returns_404(api_client):
    """22. Test POST /alerts/{alert_id}/review returns 404 for nonexistent alert ID."""
    review_payload = {
        "status": "CLOSED",
        "notes": "Testing unknown alert ID.",
        "analyst_id": "ANALYST-01",
    }
    res = api_client.post("/alerts/NONEXISTENT-ALERT-ID/review", json=review_payload)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_get_metrics(api_client, valid_transaction_payload):
    """23. Test GET /metrics returning system operational metrics derived from DB."""
    # Post a transaction
    api_client.post("/predict", json=valid_transaction_payload)

    metrics_res = api_client.get("/metrics")
    assert metrics_res.status_code == 200
    data = metrics_res.json()
    assert "total_evaluations" in data
    assert data["total_evaluations"] >= 1
    assert "evaluations_by_tier" in data


def test_get_health(api_client):
    """24. Test GET /health reporting system status and dependency readiness."""
    health_res = api_client.get("/health")
    assert health_res.status_code == 200
    data = health_res.json()
    assert "status" in data
    assert data["database"] == "connected"
    assert "model_artifacts" in data
