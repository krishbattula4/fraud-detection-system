"""System performance and drift monitoring schemas."""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class SystemMetricsResponse(BaseModel):
    """System evaluation and operational health metrics."""
    total_evaluations: int = Field(..., ge=0, description="Total transactions evaluated")
    total_alerts_triggered: int = Field(..., ge=0, description="Total high-risk alerts generated")
    avg_latency_ms: float = Field(..., ge=0.0, description="Average inference latency in milliseconds")
    active_model_version: str = Field(..., description="Currently active model version tag")
    evaluations_by_tier: Dict[str, int] = Field(..., description="Count of evaluations per risk level")


class DriftReport(BaseModel):
    """Drift monitoring summary report."""
    feature_drift_scores: Dict[str, float] = Field(..., description="Drift metric per feature (e.g. KS-statistic or PSI)")
    risk_score_drift: float = Field(..., description="Drift metric for final risk score distribution")
    drift_detected: bool = Field(..., description="Flag indicating if drift exceeds threshold")
    evaluated_at: str = Field(..., description="Timestamp of drift calculation")
