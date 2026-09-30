"""Prediction router endpoint for credit-card fraud risk evaluation."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.schemas.prediction import RiskPredictionRequest, RiskPredictionResponse
from src.db.session import get_db_session
from src.engine.prediction_service import PredictionService
from src.core.exceptions import ModelNotFoundError, ValidationError, InferenceError, RiskEngineError, DatabaseError
from src.core.logging import get_logger

logger = get_logger("predict_router")

router = APIRouter(tags=["Inference"])

# Singleton service instance
_prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Dependency injector for PredictionService instance."""
    return _prediction_service


@router.post(
    "/predict",
    response_model=RiskPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate credit-card transaction risk score",
)
def predict_transaction_risk(
    request: RiskPredictionRequest,
    db: Session = Depends(get_db_session),
    service: PredictionService = Depends(get_prediction_service),
):
    """Endpoint for evaluating transaction fraud risk.
    
    Processes incoming transaction features through the preprocessing pipeline,
    executes XGBoost, Isolation Forest, and LOF model inferences, aggregates risk
    signals into a 0-100 score, and persists transaction/prediction/alert entities.
    """
    try:
        response = service.predict_transaction_risk(
            transaction_input=request.transaction,
            db=db,
        )
        return response

    except ModelNotFoundError as e:
        logger.error(f"Model artifact missing during prediction request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {e.message}",
        )
    except ValidationError as e:
        logger.error(f"Validation failure during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Transaction validation error: {e.message}",
        )
    except (InferenceError, RiskEngineError) as e:
        logger.error(f"Inference failure during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {e.message}",
        )
    except DatabaseError as e:
        logger.error(f"Database error during prediction persistence: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Persistence error: {e.message}",
        )
