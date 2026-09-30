"""Transaction query router endpoints."""
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.db.session import get_db_session
from src.db.repository import TransactionRepository, PredictionRepository
from src.db.models import TransactionRecord
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger("transactions_router")

router = APIRouter(prefix="/transactions", tags=["Transactions"])


def _serialize_transaction(tx: TransactionRecord, pred_repo: Optional[PredictionRepository] = None) -> Dict[str, Any]:
    """Format TransactionRecord ORM entity with linked prediction info."""
    created_at_str = tx.created_at.isoformat() if isinstance(tx.created_at, datetime) else str(tx.created_at)

    latest_pred = None
    if tx.predictions and len(tx.predictions) > 0:
        latest_pred = tx.predictions[0]
    elif pred_repo:
        latest_pred = pred_repo.get_by_transaction_record_id(tx.id)

    pred_data = None
    if latest_pred:
        evaluated_at_str = latest_pred.evaluated_at.isoformat() if isinstance(latest_pred.evaluated_at, datetime) else str(latest_pred.evaluated_at)
        pred_data = {
            "prediction_id": latest_pred.id,
            "risk_score": latest_pred.risk_score,
            "risk_level": latest_pred.risk_level,
            "decision": latest_pred.decision,
            "xgboost_prob": latest_pred.xgboost_prob,
            "iforest_anomaly": latest_pred.iforest_anomaly,
            "lof_anomaly": latest_pred.lof_anomaly,
            "explanations": latest_pred.explanations or [],
            "model_version": latest_pred.model_version,
            "evaluated_at": evaluated_at_str,
        }

    return {
        "id": tx.id,
        "transaction_id": tx.transaction_id,
        "time": tx.time,
        "amount": tx.amount,
        "raw_features": tx.raw_features,
        "created_at": created_at_str,
        "prediction": pred_data,
    }


@router.get("", status_code=status.HTTP_200_OK)
def list_transactions(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_session),
):
    """Retrieve evaluated transaction history with pagination."""
    try:
        repo = TransactionRepository(db)
        pred_repo = PredictionRepository(db)
        transactions = repo.list(limit=limit, offset=offset)
        return [_serialize_transaction(tx, pred_repo) for tx in transactions]
    except DatabaseError as e:
        logger.error(f"Error listing transactions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {e.message}",
        )


@router.get("/{transaction_id}", status_code=status.HTTP_200_OK)
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db_session),
):
    """Retrieve details for a specific transaction by system primary key ID or client transaction_id."""
    try:
        repo = TransactionRepository(db)
        pred_repo = PredictionRepository(db)

        # First check primary key ID, then client transaction_id
        tx = repo.get_by_id(transaction_id)
        if not tx:
            tx = repo.get_by_transaction_id(transaction_id)

        if not tx:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction with ID '{transaction_id}' not found.",
            )

        return _serialize_transaction(tx, pred_repo)

    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Error fetching transaction '{transaction_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {e.message}",
        )
