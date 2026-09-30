"""System metrics router endpoint querying operational database stats."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.schemas.metrics import SystemMetricsResponse
from src.db.session import get_db_session
from src.db.repository import MetricsRepository
from src.core.config import get_settings
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger("metrics_router")

router = APIRouter(prefix="/metrics", tags=["Monitoring"])


@router.get("", response_model=SystemMetricsResponse, status_code=status.HTTP_200_OK)
def get_system_metrics(db: Session = Depends(get_db_session)):
    """Retrieve operational system performance metrics derived from persisted evaluation records."""
    try:
        repo = MetricsRepository(db)
        metrics_data = repo.get_operational_metrics()

        return SystemMetricsResponse(
            total_evaluations=metrics_data["total_evaluations"],
            total_alerts_triggered=metrics_data["total_alerts_triggered"],
            avg_latency_ms=0.0,
            active_model_version="1.0.0",
            evaluations_by_tier=metrics_data["evaluations_by_tier"],
        )
    except DatabaseError as e:
        logger.error(f"Error querying system metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failure: {e.message}",
        )
