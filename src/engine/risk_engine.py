"""Hybrid Risk Engine implementation combining supervised fraud probability and unsupervised anomaly signals."""
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, NamedTuple, Tuple, Optional
import numpy as np

from src.schemas.prediction import RiskLevel, DecisionAction, ModelRiskComponents
from src.core.exceptions import RiskEngineError
from src.core.logging import get_logger

logger = get_logger("risk_engine")


class RiskEngineOutput(NamedTuple):
    """Container for calculated hybrid risk output."""
    risk_score: float
    risk_level: RiskLevel
    decision: DecisionAction
    components: ModelRiskComponents


class BaseHybridRiskEngine(ABC):
    """Abstract interface for aggregating model signals into a unified 0-100 risk score.
    
    Combines:
    1. Calibrated XGBoost fraud probability [0, 1]
    2. Normalized Isolation Forest anomaly score [0, 1]
    3. Normalized LOF anomaly score [0, 1]
    """

    @abstractmethod
    def calculate_risk(
        self,
        xgboost_prob: float,
        iforest_anomaly: float,
        lof_anomaly: float,
    ) -> RiskEngineOutput:
        """Combine model risk components into calibrated 0-100 risk score and tier classification."""
        pass

    @abstractmethod
    def classify_risk_tier(self, risk_score: float) -> Tuple[RiskLevel, DecisionAction]:
        """Map 0-100 risk score to RiskLevel tier and DecisionAction."""
        pass


class HybridRiskEngine(BaseHybridRiskEngine):
    """Concrete hybrid risk scoring engine.
    
    Combines calibrated XGBoost probability with normalized Isolation Forest and LOF anomaly scores.
    The resulting score is scaled from 0.0 to 100.0 and categorized into configurable risk tiers.
    
    NOTE: The 0-100 risk score and risk bands are project-specific decision-support representations
    and are not universal industry or banking regulatory standards.
    """

    DEFAULT_WEIGHTS = {
        "xgboost": 0.60,
        "isolation_forest": 0.25,
        "lof": 0.15,
    }

    DEFAULT_RISK_BANDS = [
        (0.0, 39.9999, RiskLevel.LOW, DecisionAction.APPROVE),
        (40.0, 69.9999, RiskLevel.MEDIUM, DecisionAction.FLAG_FOR_REVIEW),
        (70.0, 89.9999, RiskLevel.HIGH, DecisionAction.FLAG_FOR_REVIEW),
        (90.0, 100.0, RiskLevel.CRITICAL, DecisionAction.DECLINE),
    ]

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        risk_bands: Optional[list] = None,
    ):
        self.weights = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
        self.risk_bands = risk_bands if risk_bands is not None else list(self.DEFAULT_RISK_BANDS)
        self._validate_weights(self.weights)

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Ensure weights are non-negative, finite, and sum to approximately 1.0."""
        required_keys = {"xgboost", "isolation_forest", "lof"}
        if not required_keys.issubset(weights.keys()):
            raise RiskEngineError(f"Weights dict missing required keys. Must include: {required_keys}")

        total_weight = 0.0
        for key, val in weights.items():
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise RiskEngineError(f"Weight for '{key}' must be a finite number, got {val}")
            if val < 0.0:
                raise RiskEngineError(f"Weight for '{key}' cannot be negative, got {val}")
            total_weight += val

        if not math.isclose(total_weight, 1.0, abs_tol=1e-4):
            raise RiskEngineError(f"Model risk component weights must sum to 1.0, got sum = {total_weight:.4f}")

    def _validate_input_component(self, name: str, val: float) -> float:
        """Validate an individual component score is a finite float in [0.0, 1.0]."""
        if not isinstance(val, (int, float, np.number)) or not math.isfinite(val):
            raise RiskEngineError(f"Input '{name}' must be a finite float, got {val}")
        
        float_val = float(val)
        if float_val < 0.0 or float_val > 1.0:
            raise RiskEngineError(f"Input '{name}' must be within [0.0, 1.0], got {float_val}")

        return float_val

    def calculate_risk(
        self,
        xgboost_prob: float,
        iforest_anomaly: float,
        lof_anomaly: float,
    ) -> RiskEngineOutput:
        """Calculate aggregated 0-100 hybrid risk score from component signals.
        
        Validation:
        - Validates all input components are finite floats in [0, 1]
        - Calculates weighted aggregation
        - Scales to 0-100 bounded score
        - Maps score to RiskLevel tier and DecisionAction
        """
        xgb_val = self._validate_input_component("xgboost_prob", xgboost_prob)
        if_val = self._validate_input_component("iforest_anomaly", iforest_anomaly)
        lof_val = self._validate_input_component("lof_anomaly", lof_anomaly)

        raw_weighted_score = (
            self.weights["xgboost"] * xgb_val
            + self.weights["isolation_forest"] * if_val
            + self.weights["lof"] * lof_val
        )

        # Scale to 0.0 - 100.0 and bound explicitly
        score_100 = raw_weighted_score * 100.0
        score_bounded = float(np.clip(score_100, 0.0, 100.0))
        risk_score = round(score_bounded, 2)

        risk_level, decision = self.classify_risk_tier(risk_score)

        components = ModelRiskComponents(
            xgboost_calibrated_prob=xgb_val,
            isolation_forest_anomaly_score=if_val,
            lof_anomaly_score=lof_val,
        )

        logger.debug(
            f"Calculated risk score: {risk_score:.2f} (level={risk_level.value}, decision={decision.value})"
        )

        return RiskEngineOutput(
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision,
            components=components,
        )

    def classify_risk_tier(self, risk_score: float) -> Tuple[RiskLevel, DecisionAction]:
        """Map a 0-100 risk score to a RiskLevel tier and DecisionAction."""
        if not isinstance(risk_score, (int, float, np.number)) or not math.isfinite(risk_score):
            raise RiskEngineError(f"Risk score must be a finite number, got {risk_score}")

        score = float(risk_score)

        for min_val, max_val, tier, decision in self.risk_bands:
            if min_val <= score <= max_val:
                return tier, decision

        # Boundary fallback for floating point edge cases
        if score < 0.0:
            return RiskLevel.LOW, DecisionAction.APPROVE
        return RiskLevel.CRITICAL, DecisionAction.DECLINE
