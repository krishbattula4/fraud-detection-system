"""Integration tests for POST /predict/advanced endpoint, router isolation, real artifact inference, validation, and DB persistence."""
import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.app import app
from src.db.session import get_db_session
from src.db.repository import TransactionRepository, PredictionRepository, AlertRepository
from src.ml.advanced_models import get_advanced_artifact_dir
from src.core.exceptions import ModelNotFoundError

client = TestClient(app)


def test_post_predict_advanced_real_artifacts():
    """Verify POST /predict/advanced executes end-to-end inference with real IEEE-CIS joblib artifacts."""
    adv_dir = get_advanced_artifact_dir()
    xgb_path = os.path.join(adv_dir, "advanced_xgboost_v2.0.0.joblib")

    if not os.path.exists(xgb_path):
        pytest.skip("Authentic IEEE-CIS joblib artifacts not present.")

    payload = {
        "transaction_id": "TX-API-ADV-1001",
        "timestamp": 100000.0,
        "amount": 350.0,
        "card_id": "card_5500",
        "billing_region": "addr_200",
        "purchaser_email_domain": "gmail.com",
        "device_type": "mobile",
        "counting_features": {"C1": 2.0, "C2": 4.0},
        "timedelta_features": {"D1": 10.0},
    }

    response = client.post("/predict/advanced", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "TX-API-ADV-1001"
    assert data["prediction_id"].startswith("adv-pred-")
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data["decision"] in ["APPROVE", "FLAG_FOR_REVIEW", "DECLINE"]
    assert data["model_version"] == "2.0.0-authentic-ieee"
    assert "supervised_xgboost_prob" in data["model_components"]
    assert "isolation_forest_anomaly_score" in data["model_components"]
    assert "behavioral_velocity_score" in data["model_components"]
    assert isinstance(data["rule_triggers"], list)
    assert isinstance(data["explanations"], list)


def test_post_predict_advanced_validation_error():
    """Verify malformed or invalid advanced transaction input returns HTTP 422 Unprocessable Entity."""
    # Missing required 'card_id'
    invalid_payload = {
        "transaction_id": "TX-API-ADV-ERR",
        "timestamp": 1000.0,
        "amount": 100.0,
    }

    response = client.post("/predict/advanced", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()

    # Negative amount
    negative_amt_payload = {
        "transaction_id": "TX-API-ADV-ERR-2",
        "timestamp": 1000.0,
        "amount": -50.0,
        "card_id": "card_1001",
    }
    response_neg = client.post("/predict/advanced", json=negative_amt_payload)
    assert response_neg.status_code == 422


def test_router_isolation_baseline_vs_advanced():
    """Verify router isolation: baseline /predict and advanced /predict/advanced operate independently."""
    # 1. Baseline Request
    baseline_payload = {
        "transaction": {
            "transaction_id": "TX-BASE-TEST-001",
            "time": 100.0,
            "amount": 50.0,
            **{f"v{i}": 0.1 * i for i in range(1, 29)},
        }
    }

    resp_base = client.post("/predict", json=baseline_payload)
    assert resp_base.status_code == 200
    base_data = resp_base.json()
    assert base_data["model_version"] == "1.0.0"
    assert "xgboost_calibrated_prob" in base_data["model_components"]

    # 2. Advanced Request
    adv_dir = get_advanced_artifact_dir()
    xgb_path = os.path.join(adv_dir, "advanced_xgboost_v2.0.0.joblib")
    if not os.path.exists(xgb_path):
        pytest.skip("Authentic IEEE-CIS joblib artifacts not present.")

    advanced_payload = {
        "transaction_id": "TX-ISO-TEST-001",
        "timestamp": 50000.0,
        "amount": 125.0,
        "card_id": "card_7777",
    }

    resp_adv = client.post("/predict/advanced", json=advanced_payload)
    assert resp_adv.status_code == 200
    adv_data = resp_adv.json()
    assert adv_data["model_version"] == "2.0.0-authentic-ieee"
    assert "supervised_xgboost_prob" in adv_data["model_components"]


def test_advanced_predict_model_missing_error():
    """Verify missing advanced model artifact raises ModelNotFoundError returning HTTP 503."""
    with patch(
        "src.engine.advanced_prediction_service.AdvancedPredictionService.predict_advanced_risk",
        side_effect=ModelNotFoundError("Advanced XGBoost artifact missing"),
    ):
        payload = {
            "transaction_id": "TX-MISSING-TEST",
            "timestamp": 1000.0,
            "amount": 100.0,
            "card_id": "card_123",
        }
        response = client.post("/predict/advanced", json=payload)
        assert response.status_code == 503
        assert "Advanced model service unavailable" in response.json()["detail"]
