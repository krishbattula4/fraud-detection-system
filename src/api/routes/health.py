"""Health check router endpoint reporting system dependency status."""
import os
from typing import Dict, Any
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.session import get_db_session
from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger("health_router")

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db_session)) -> Dict[str, Any]:
    """Health check endpoint evaluating database connectivity and model artifact readiness."""
    settings = get_settings()
    
    # 1. Database Check
    db_status = "unavailable"
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")

    # 2. Model Artifacts Check
    model_dir = settings.model_dir
    artifacts_status = "unavailable"
    if os.path.exists(model_dir):
        joblib_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
        if joblib_files:
            artifacts_status = "available"

    overall_status = "healthy" if db_status == "connected" and artifacts_status == "available" else "degraded"

    return {
        "status": overall_status,
        "service": "fraud-detection-system",
        "version": "0.1.0",
        "database": db_status,
        "model_artifacts": artifacts_status,
        "model_dir": model_dir,
    }
