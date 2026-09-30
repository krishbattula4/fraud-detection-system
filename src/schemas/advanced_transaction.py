"""Advanced multi-entity transaction schemas supporting IEEE-CIS fraud detection features."""
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from src.schemas.prediction import RiskLevel, DecisionAction

class AdvancedTransactionInput(BaseModel):
    """API payload for multi-entity advanced transaction risk scoring."""
    transaction_id: str = Field(..., description="Unique client transaction identifier")
    timestamp: float = Field(..., ge=0.0, description="Elapsed time in seconds (TransactionDT or Unix timestamp)")
    amount: float = Field(..., ge=0.0, description="Transaction amount in USD")
    card_id: str = Field(..., description="Card issuer / account identifier (e.g. card1)")
    billing_region: Optional[str] = Field(default=None, description="Billing region or country code (e.g. addr1)")
    purchaser_email_domain: Optional[str] = Field(default=None, description="Purchaser email domain (P_emaildomain)")
    recipient_email_domain: Optional[str] = Field(default=None, description="Recipient email domain (R_emaildomain)")
    product_category: Optional[str] = Field(default="W", description="Product transaction category code (ProductCD)")
    card_network: Optional[str] = Field(default="visa", description="Card network brand (card4)")
    card_type: Optional[str] = Field(default="debit", description="Card funding type (card6)")
    device_type: Optional[str] = Field(default=None, description="Device channel type (desktop, mobile)")
    device_info: Optional[str] = Field(default=None, description="Device OS/Browser details (DeviceInfo)")
    counting_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Pre-aggregated count signals (C1-C14)")
    timedelta_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Timedelta signals (D1-D15)")


class AdvancedModelRiskComponents(BaseModel):
    """Multi-model risk signals for advanced transaction pipeline."""
    supervised_xgboost_prob: float = Field(..., ge=0.0, le=1.0, description="Calibrated XGBoost fraud probability")
    isolation_forest_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized Isolation Forest anomaly score")
    behavioral_velocity_score: float = Field(..., ge=0.0, le=1.0, description="Normalized entity velocity risk score")


class AdvancedRiskPredictionResponse(BaseModel):
    """Complete response payload for advanced risk intelligence platform."""
    prediction_id: str = Field(..., description="Unique prediction UUID")
    transaction_id: str = Field(..., description="Client transaction ID")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Calibrated hybrid risk score (0 to 100)")
    risk_level: RiskLevel = Field(..., description="Risk tier classification")
    decision: DecisionAction = Field(..., description="Automated decision action")
    model_components: AdvancedModelRiskComponents = Field(..., description="Component risk signal breakdown")
    behavioral_signals: Dict[str, Any] = Field(default_factory=dict, description="Extracted velocity & novelty features")
    rule_triggers: List[str] = Field(default_factory=list, description="Triggered deterministic risk rules")
    explanations: List[Dict[str, Any]] = Field(default_factory=list, description="Top feature importance attributions")
    model_version: str = Field(default="v2.0.0-advanced", description="Engine model version")
    evaluated_at: str = Field(..., description="ISO 8601 evaluation timestamp")
