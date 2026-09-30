"""Advanced prediction router endpoint for multi-entity IEEE-CIS fraud risk evaluation."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.schemas.advanced_transaction import AdvancedTransactionInput, AdvancedRiskPredictionResponse
from src.db.session import get_db_session
from src.engine.advanced_prediction_service import AdvancedPredictionService
from src.core.exceptions import ModelNotFoundError, ValidationError, InferenceError, RiskEngineError, DatabaseError
from src.core.logging import get_logger

logger = get_logger("advanced_predict_router")

router = APIRouter(tags=["Advanced Inference"])

# Singleton advanced prediction service instance
_advanced_prediction_service = AdvancedPredictionService()


def get_advanced_prediction_service() -> AdvancedPredictionService:
    """Dependency injector for AdvancedPredictionService instance."""
    return _advanced_prediction_service


@router.post(
    "/predict/advanced",
    response_model=AdvancedRiskPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate multi-entity advanced transaction risk score",
)
def predict_advanced_transaction_risk(
    request: AdvancedTransactionInput,
    db: Session = Depends(get_db_session),
    service: AdvancedPredictionService = Depends(get_advanced_prediction_service),
):
    """Endpoint for evaluating multi-entity advanced transaction fraud risk.
    
    Processes multi-entity input features (amount, timestamp, card_id, velocity, email domain, device),
    executes authentic IEEE-CIS XGBoost and Isolation Forest model inferences, aggregates risk signals
    into a 0-100 score, generates evidence-based explanations, and persists transaction/prediction/alert entities.
    """
    try:
        response = service.predict_advanced_risk(
            transaction_input=request,
            db=db,
        )
        return response

    except ModelNotFoundError as e:
        logger.error(f"Advanced model artifact missing during prediction request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Advanced model service unavailable: {e.message}",
        )
    except ValidationError as e:
        logger.error(f"Validation failure during advanced prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Advanced transaction validation error: {e.message}",
        )
    except (InferenceError, RiskEngineError) as e:
        logger.error(f"Inference failure during advanced prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced inference pipeline error: {e.message}",
        )
    except DatabaseError as e:
        logger.error(f"Database error during advanced prediction persistence: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced persistence error: {e.message}",
        )
