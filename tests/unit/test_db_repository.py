"""Unit test suite for Phase 4 SQLAlchemy database repositories."""
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.session import Base
from src.db.models import TransactionRecord, PredictionRecord, AlertRecord, ReviewRecord
from src.db.repository import (
    TransactionRepository,
    PredictionRepository,
    AlertRepository,
    ReviewRepository,
    MetricsRepository,
)
from src.schemas.alert import AlertStatus
from src.schemas.prediction import RiskLevel, DecisionAction


@pytest.fixture
def db_session():
    """In-memory SQLite database session fixture."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_transaction_persistence(db_session):
    """11. Test persisting and retrieving TransactionRecord."""
    repo = TransactionRepository(db_session)
    tx_record = TransactionRecord(
        id="tx-001",
        transaction_id="CLIENT-TX-100",
        time=100.0,
        amount=250.50,
        raw_features={"Time": 100.0, "V1": 0.5, "Amount": 250.50},
        created_at=datetime.now(timezone.utc),
    )

    saved_tx = repo.add(tx_record)
    assert saved_tx.id == "tx-001"

    retrieved = repo.get_by_id("tx-001")
    assert retrieved is not None
    assert retrieved.transaction_id == "CLIENT-TX-100"
    assert retrieved.amount == 250.50

    by_client_id = repo.get_by_transaction_id("CLIENT-TX-100")
    assert by_client_id is not None
    assert by_client_id.id == "tx-001"


def test_prediction_persistence(db_session):
    """12. Test persisting PredictionRecord linked to TransactionRecord."""
    tx_repo = TransactionRepository(db_session)
    pred_repo = PredictionRepository(db_session)

    tx_record = tx_repo.add(
        TransactionRecord(
            id="tx-002",
            transaction_id="CLIENT-TX-101",
            time=200.0,
            amount=50.0,
            raw_features={"Time": 200.0, "Amount": 50.0},
        )
    )

    pred_record = PredictionRecord(
        id="pred-001",
        transaction_record_id=tx_record.id,
        risk_score=85.5,
        risk_level=RiskLevel.HIGH.value,
        decision=DecisionAction.FLAG_FOR_REVIEW.value,
        xgboost_prob=0.80,
        iforest_anomaly=0.75,
        lof_anomaly=0.60,
        explanations=[{"feature_name": "Amount", "feature_value": 50.0, "importance_score": 0.35}],
        model_version="1.0.0",
        evaluated_at=datetime.now(timezone.utc),
    )

    saved_pred = pred_repo.add(pred_record)
    assert saved_pred.id == "pred-001"

    retrieved_pred = pred_repo.get_by_id("pred-001")
    assert retrieved_pred is not None
    assert retrieved_pred.risk_score == 85.5
    assert retrieved_pred.transaction_record_id == "tx-002"


def test_alert_persistence(db_session):
    """13. Test persisting and filtering AlertRecord."""
    alert_repo = AlertRepository(db_session)

    alert1 = alert_repo.add(
        AlertRecord(
            id="alt-001",
            prediction_id="pred-001",
            risk_score=75.0,
            risk_level=RiskLevel.HIGH.value,
            status=AlertStatus.OPEN.value,
        )
    )

    alert2 = alert_repo.add(
        AlertRecord(
            id="alt-002",
            prediction_id="pred-002",
            risk_score=95.0,
            risk_level=RiskLevel.CRITICAL.value,
            status=AlertStatus.CONFIRMED_FRAUD.value,
        )
    )

    all_alerts = alert_repo.list()
    assert len(all_alerts) == 2

    open_alerts = alert_repo.list(status=AlertStatus.OPEN.value)
    assert len(open_alerts) == 1
    assert open_alerts[0].id == "alt-001"

    high_score_alerts = alert_repo.list(min_risk_score=80.0)
    assert len(high_score_alerts) == 1
    assert high_score_alerts[0].id == "alt-002"


def test_alert_review_update(db_session):
    """14. Test updating alert review status and logging audit ReviewRecord."""
    alert_repo = AlertRepository(db_session)
    review_repo = ReviewRepository(db_session)

    alert = alert_repo.add(
        AlertRecord(
            id="alt-003",
            prediction_id="pred-003",
            risk_score=88.0,
            risk_level=RiskLevel.HIGH.value,
            status=AlertStatus.OPEN.value,
        )
    )

    review_rec = review_repo.add(
        ReviewRecord(
            id="rev-001",
            alert_id=alert.id,
            analyst_id="ANALYST-99",
            previous_status=alert.status,
            new_status=AlertStatus.CLOSED.value,
            notes="Transaction confirmed legitimate after cardholder call.",
        )
    )
    assert review_rec.id == "rev-001"

    updated_alert = alert_repo.update_review(
        alert=alert,
        new_status=AlertStatus.CLOSED.value,
        reviewer_notes="Transaction confirmed legitimate after cardholder call.",
    )

    assert updated_alert.status == AlertStatus.CLOSED.value
    assert updated_alert.reviewed_at is not None
    assert "legitimate" in updated_alert.reviewer_notes


def test_relationship_integrity(db_session):
    """15. Test SQLAlchemy relationship integrity across Transaction, Prediction, Alert, and Review."""
    tx_repo = TransactionRepository(db_session)
    pred_repo = PredictionRepository(db_session)
    alert_repo = AlertRepository(db_session)
    review_repo = ReviewRepository(db_session)

    tx = tx_repo.add(TransactionRecord(id="tx-100", time=1.0, amount=10.0, raw_features={}))
    pred = pred_repo.add(
        PredictionRecord(
            id="pred-100",
            transaction_record_id=tx.id,
            risk_score=92.0,
            risk_level="CRITICAL",
            decision="DECLINE",
            xgboost_prob=0.9,
            iforest_anomaly=0.8,
            lof_anomaly=0.7,
            model_version="1.0.0",
        )
    )
    alt = alert_repo.add(AlertRecord(id="alt-100", prediction_id=pred.id, risk_score=92.0, risk_level="CRITICAL", status="OPEN"))
    rev = review_repo.add(ReviewRecord(id="rev-100", alert_id=alt.id, analyst_id="A1", previous_status="OPEN", new_status="CLOSED"))

    # Refresh alert to test backref relationships
    fetched_alert = alert_repo.get_by_id("alt-100")
    assert fetched_alert.prediction is not None
    assert fetched_alert.prediction.transaction is not None
    assert fetched_alert.prediction.transaction.id == "tx-100"
    assert len(fetched_alert.reviews) == 1
    assert fetched_alert.reviews[0].analyst_id == "A1"
