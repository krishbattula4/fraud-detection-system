"""Alert and investigation review schemas."""
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from src.schemas.prediction import RiskLevel, DecisionAction, ModelRiskComponents


class AlertStatus(str, Enum):
    """Investigation status of a flagged high-risk alert."""
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    CLOSED = "CLOSED"


class AlertFilter(BaseModel):
    """Filter parameters for querying risk alerts."""
    status: Optional[AlertStatus] = Field(default=None, description="Filter by alert status")
    min_risk_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Minimum risk score boundary")
    risk_level: Optional[RiskLevel] = Field(default=None, description="Filter by risk tier")
    limit: int = Field(default=50, ge=1, le=1000, description="Max record limit")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class AlertResponse(BaseModel):
    """Risk alert entity response payload."""
    alert_id: str = Field(..., description="Unique alert identifier")
    prediction_id: str = Field(..., description="Associated risk prediction ID")
    transaction_id: Optional[str] = Field(default=None, description="Client transaction ID")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Calculated 0-100 risk score")
    risk_level: RiskLevel = Field(..., description="Risk tier")
    status: AlertStatus = Field(..., description="Current alert status")
    created_at: str = Field(..., description="ISO 8601 timestamp of alert creation")
    reviewed_at: Optional[str] = Field(default=None, description="ISO 8601 timestamp of last analyst review")
    reviewer_notes: Optional[str] = Field(default=None, description="Analyst investigation notes")


class AlertReviewRequest(BaseModel):
    """Request payload for updating an alert investigation status."""
    status: AlertStatus = Field(..., description="Updated investigation status")
    notes: Optional[str] = Field(default=None, max_length=1000, description="Analyst review notes")
    analyst_id: str = Field(..., description="ID of reviewing analyst")
