"""Phase 5 End-to-End Integration Test Suite verifying Simulator -> API Client -> FastAPI -> DB path."""
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
from src.dashboard.api_client import DashboardAPIClient
from src.simulation.simulator import TransactionSimulator
from src.data.preprocessor import DataPreprocessor, FEATURE_COLUMNS
from src.ml.supervised import XGBoostFraudModel
from src.ml.anomaly import IsolationForestAnomalyDetector, LOFAnomalyDetector
from src.ml.calibration import ProbabilityCalibrator, MinMaxScoreNormalizer
from src.explainability.explainer import SHAPTransactionExplainer
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
def integration_api_client(test_db_engine, mock_prediction_service):
    """FastAPI app TestClient coupled with DashboardAPIClient via custom transport adapter."""
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

    # Construct DashboardAPIClient using TestClient transport
    client = DashboardAPIClient(base_url="http://testserver")

    # Monkeypatch DashboardAPIClient._request to route through FastAPI TestClient
    test_http_client = TestClient(app)

    def _test_request(method: str, endpoint: str, json=None, params=None):
        res = test_http_client.request(method=method, url=endpoint, json=json, params=params)
        if res.status_code == 200:
            return {"success": True, "data": res.json(), "status_code": 200}
        error_detail = res.json().get("detail", res.text) if res.headers.get("content-type") == "application/json" else res.text
        return {"success": False, "error": error_detail, "status_code": res.status_code}

    client._request = _test_request
    return client


def test_full_phase5_integration_flow(integration_api_client):
    """Verify full end-to-end Phase 5 path: Simulator -> DashboardAPIClient -> FastAPI -> DB -> Review -> Metrics."""
    # 1. Simulator generates test transaction vector
    sim = TransactionSimulator(seed=999)
    test_tx = sim.generate_test_transaction(transaction_id="SIM-PHASE5-TEST-100", is_anomaly=True)
    payload = test_tx.model_dump()

    # 2. Dashboard API Client submits payload to POST /predict
    pred_res = integration_api_client.predict_transaction(payload)
    assert pred_res["success"] is True
    pred_data = pred_res["data"]
    assert pred_data["transaction_id"] == "SIM-PHASE5-TEST-100"
    assert 0.0 <= pred_data["risk_score"] <= 100.0

    # 3. Retrieve stored transaction info via GET /transactions/{id}
    tx_res = integration_api_client.get_transaction("SIM-PHASE5-TEST-100")
    assert tx_res["success"] is True
    tx_data = tx_res["data"]
    assert tx_data["transaction_id"] == "SIM-PHASE5-TEST-100"
    assert tx_data["prediction"] is not None

    # 4. Retrieve alert queue via GET /alerts
    alerts_res = integration_api_client.get_alerts()
    assert alerts_res["success"] is True
    alerts = alerts_res["data"]
    assert len(alerts) >= 1
    alert_id = alerts[0]["alert_id"]

    # 5. Analyst submits review via POST /alerts/{id}/review
    review_res = integration_api_client.review_alert(
        alert_id=alert_id,
        new_status="CONFIRMED_FRAUD",
        notes="Phase 5 integration verification passed.",
        analyst_id="ANALYST-TEST-01",
    )
    assert review_res["success"] is True
    assert review_res["data"]["status"] == "CONFIRMED_FRAUD"

    # 6. Verify GET /metrics reflects evaluated transaction count
    metrics_res = integration_api_client.get_metrics()
    assert metrics_res["success"] is True
    assert metrics_res["data"]["total_evaluations"] >= 1

    # 7. Verify GET /health
    health_res = integration_api_client.get_health()
    assert health_res["success"] is True
    assert health_res["data"]["database"] == "connected"
