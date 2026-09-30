"""Alert management and analyst review router endpoints."""
import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.schemas.alert import AlertResponse, AlertReviewRequest, AlertStatus, AlertFilter
from src.schemas.prediction import RiskLevel
from src.db.session import get_db_session
from src.db.repository import AlertRepository, ReviewRepository, PredictionRepository, TransactionRepository
from src.db.models import AlertRecord, ReviewRecord
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger("alerts_router")

router = APIRouter(prefix="/alerts", tags=["Alerts"])


def _build_alert_response(alert: AlertRecord) -> AlertResponse:
    """Map ORM AlertRecord to Pydantic AlertResponse schema."""
    tx_client_id = None
    if alert.prediction and alert.prediction.transaction:
        tx_client_id = alert.prediction.transaction.transaction_id

    created_at_str = alert.created_at.isoformat() if isinstance(alert.created_at, datetime) else str(alert.created_at)
    reviewed_at_str = alert.reviewed_at.isoformat() if alert.reviewed_at and isinstance(alert.reviewed_at, datetime) else (str(alert.reviewed_at) if alert.reviewed_at else None)

    return AlertResponse(
        alert_id=alert.id,
        prediction_id=alert.prediction_id,
        transaction_id=tx_client_id,
        risk_score=alert.risk_score,
        risk_level=RiskLevel(alert.risk_level),
        status=AlertStatus(alert.status),
        created_at=created_at_str,
        reviewed_at=reviewed_at_str,
        reviewer_notes=alert.reviewer_notes,
    )


@router.get("", response_model=List[AlertResponse], status_code=status.HTTP_200_OK)
def list_alerts(
    status_filter: Optional[AlertStatus] = Query(default=None, alias="status"),
    min_risk_score: Optional[float] = Query(default=None, ge=0.0, le=100.0),
    risk_level: Optional[RiskLevel] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_session),
):
    """Retrieve risk alerts with optional status, min risk score, and risk level filtering."""
    try:
        repo = AlertRepository(db)
        status_val = status_filter.value if status_filter else None
        level_val = risk_level.value if risk_level else None

        alerts = repo.list(
            status=status_val,
            min_risk_score=min_risk_score,
            risk_level=level_val,
            limit=limit,
            offset=offset,
        )
        return [_build_alert_response(alt) for alt in alerts]
    except DatabaseError as e:
        logger.error(f"Error querying alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failure: {e.message}",
        )


@router.post("/{alert_id}/review", response_model=AlertResponse, status_code=status.HTTP_200_OK)
def review_alert(
    alert_id: str,
    review_payload: AlertReviewRequest,
    db: Session = Depends(get_db_session),
):
    """Submit analyst investigation review for a flagged risk alert."""
    try:
        alert_repo = AlertRepository(db)
        review_repo = ReviewRepository(db)

        alert = alert_repo.get_by_id(alert_id)
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert with ID '{alert_id}' not found.",
            )

        previous_status = alert.status
        new_status = review_payload.status.value

        # Log review record for auditability
        review_id = f"rev-{uuid.uuid4().hex[:12]}"
        now_dt = datetime.now(timezone.utc)
        review_record = ReviewRecord(
            id=review_id,
            alert_id=alert.id,
            analyst_id=review_payload.analyst_id,
            previous_status=previous_status,
            new_status=new_status,
            notes=review_payload.notes,
            reviewed_at=now_dt,
        )
        review_repo.add(review_record)

        # Update alert state
        updated_alert = alert_repo.update_review(
            alert=alert,
            new_status=new_status,
            reviewer_notes=review_payload.notes,
        )

        return _build_alert_response(updated_alert)

    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Error updating alert review for '{alert_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database update failure: {e.message}",
        )
