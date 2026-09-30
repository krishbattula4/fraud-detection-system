"""Unit tests for transaction and risk prediction Pydantic schemas."""
import pytest
from pydantic import ValidationError as PydanticValidationError
from src.schemas.transaction import TransactionInput, TransactionBatch
from src.schemas.prediction import (
    RiskPredictionRequest,
    RiskPredictionResponse,
    RiskLevel,
    DecisionAction,
    ModelRiskComponents,
)


def test_transaction_input_to_feature_vector(sample_transaction_input):
    """Verify feature vector extraction order matches pipeline expectations."""
    vec = sample_transaction_input.to_feature_vector()
    assert len(vec) == 30
    assert vec[0] == 100.0  # Time
    assert vec[-1] == 250.50  # Amount
    assert vec[1] == -1.35  # V1


def test_transaction_input_negative_amount_raises_error():
    """Verify negative transaction amount raises validation error."""
    with pytest.raises(PydanticValidationError):
        TransactionInput(
            time=10.0,
            amount=-50.0,
            v1=0.0, v2=0.0, v3=0.0, v4=0.0, v5=0.0, v6=0.0, v7=0.0, v8=0.0,
            v9=0.0, v10=0.0, v11=0.0, v12=0.0, v13=0.0, v14=0.0, v15=0.0, v16=0.0,
            v17=0.0, v18=0.0, v19=0.0, v20=0.0, v21=0.0, v22=0.0, v23=0.0, v24=0.0,
            v25=0.0, v26=0.0, v27=0.0, v28=0.0
        )


def test_model_risk_components_validation():
    """Verify ModelRiskComponents enforces bounds [0.0, 1.0]."""
    comp = ModelRiskComponents(
        xgboost_calibrated_prob=0.85,
        isolation_forest_anomaly_score=0.40,
        lof_anomaly_score=0.30,
    )
    assert comp.xgboost_calibrated_prob == 0.85

    with pytest.raises(PydanticValidationError):
        ModelRiskComponents(
            xgboost_calibrated_prob=1.5,  # Exceeds max 1.0
            isolation_forest_anomaly_score=0.4,
            lof_anomaly_score=0.3,
        )
