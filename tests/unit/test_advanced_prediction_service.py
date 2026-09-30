"""Unit tests for Advanced Prediction Service, feature extraction, and pipeline orchestration."""
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.session import Base
from src.schemas.advanced_transaction import AdvancedTransactionInput
from src.engine.advanced_prediction_service import AdvancedPredictionService
from src.schemas.prediction import RiskLevel, DecisionAction
from src.db.repository import TransactionRepository, PredictionRepository, AlertRepository
from src.ml.advanced_models import get_advanced_artifact_dir


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_advanced_prediction_service_end_to_end(in_memory_db):
    """Verify end-to-end advanced prediction service using trained joblib artifacts."""
    adv_dir = get_advanced_artifact_dir()
    xgb_path = os.path.join(adv_dir, "advanced_xgboost_v2.0.0.joblib")

    if not os.path.exists(xgb_path):
        pytest.skip("Advanced joblib artifacts not available for test execution.")

    service = AdvancedPredictionService()

    tx_input = AdvancedTransactionInput(
        transaction_id="TX-ADV-TEST-001",
        timestamp=86450.0,
        amount=250.0,
        card_id="card_1001",
        billing_region="addr_101",
        purchaser_email_domain="gmail.com",
        device_type="mobile",
        counting_features={"C1": 3.0, "C2": 5.0},
        timedelta_features={"D1": 14.0},
    )

    response = service.predict_advanced_risk(tx_input, db=in_memory_db)

    assert response.transaction_id == "TX-ADV-TEST-001"
    assert response.prediction_id.startswith("adv-pred-")
    assert 0.0 <= response.risk_score <= 100.0
    assert response.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert response.decision in [DecisionAction.APPROVE, DecisionAction.FLAG_FOR_REVIEW, DecisionAction.DECLINE]
    assert response.model_version == "2.0.0-authentic-ieee"
    assert 0.0 <= response.model_components.supervised_xgboost_prob <= 1.0
    assert 0.0 <= response.model_components.isolation_forest_anomaly_score <= 1.0

    # DB Persistence Verification
    tx_repo = TransactionRepository(in_memory_db)
    pred_repo = PredictionRepository(in_memory_db)
    alert_repo = AlertRepository(in_memory_db)

    db_tx = tx_repo.get_by_transaction_id("TX-ADV-TEST-001")
    assert db_tx is not None
    assert db_tx.amount == 250.0

    db_preds = pred_repo.list()
    assert len(db_preds) >= 1
    assert db_preds[0].model_version == "2.0.0-authentic-ieee"
