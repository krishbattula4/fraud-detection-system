"""Risk prediction request and response models."""
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from src.schemas.transaction import TransactionInput


class RiskLevel(str, Enum):
    """Categorical risk tiers based on 0-100 risk score."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DecisionAction(str, Enum):
    """Automated decision action derived from risk classification."""
    APPROVE = "APPROVE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    DECLINE = "DECLINE"


class ModelRiskComponents(BaseModel):
    """Individual model risk component scores prior to hybrid aggregation."""
    xgboost_calibrated_prob: float = Field(..., ge=0.0, le=1.0, description="Calibrated XGBoost fraud probability")
    isolation_forest_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized Isolation Forest anomaly score")
    lof_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized Local Outlier Factor anomaly score")


class RiskPredictionRequest(BaseModel):
    """API request payload for risk evaluation."""
    transaction: TransactionInput = Field(..., description="Transaction features for risk scoring")


class FeatureImportanceExplanation(BaseModel):
    """Feature impact explanation (e.g. derived from SHAP values)."""
    feature_name: str = Field(..., description="Feature identifier (e.g. Amount, V14)")
    feature_value: float = Field(..., description="Actual feature value in transaction")
    importance_score: float = Field(..., description="SHAP attribution or impact magnitude")


class RiskPredictionResponse(BaseModel):
    """Complete risk assessment response payload."""
    prediction_id: str = Field(..., description="Unique prediction UUID")
    transaction_id: Optional[str] = Field(default=None, description="Client transaction ID if provided")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Hybrid risk score scaled from 0 to 100")
    risk_level: RiskLevel = Field(..., description="Risk tier classification")
    decision: DecisionAction = Field(..., description="Automated risk decision action")
    model_components: ModelRiskComponents = Field(..., description="Individual model component risk breakdown")
    explanations: List[FeatureImportanceExplanation] = Field(default_factory=list, description="Top feature importance attributions")
    model_version: str = Field(..., description="Model version tag executing inference")
    evaluated_at: str = Field(..., description="ISO 8601 timestamp of evaluation")
