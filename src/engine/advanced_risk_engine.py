"""Advanced Hybrid Risk Engine combining supervised XGBoost, unsupervised Isolation Forest, and behavioral velocity signals."""
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, NamedTuple, Tuple, Optional, List
import numpy as np

from src.schemas.prediction import RiskLevel, DecisionAction
from src.schemas.advanced_transaction import AdvancedModelRiskComponents
from src.core.exceptions import RiskEngineError
from src.core.logging import get_logger

logger = get_logger("advanced_risk_engine")


class AdvancedRiskEngineOutput(NamedTuple):
    """Container for calculated advanced hybrid risk output."""
    risk_score: float
    risk_level: RiskLevel
    decision: DecisionAction
    components: AdvancedModelRiskComponents
    rule_triggers: List[str]
    explanations: List[Dict[str, Any]]


class BaseAdvancedHybridRiskEngine(ABC):
    """Abstract interface for aggregating advanced multi-entity signals into a 0-100 risk score."""

    @abstractmethod
    def calculate_risk(
        self,
        supervised_xgboost_prob: float,
        isolation_forest_anomaly: float,
        behavioral_velocity: float,
        feature_context: Optional[Dict[str, Any]] = None,
    ) -> AdvancedRiskEngineOutput:
        """Combine model signals and context into calibrated 0-100 risk score and tier classification."""
        pass

    @abstractmethod
    def classify_risk_tier(self, risk_score: float) -> Tuple[RiskLevel, DecisionAction]:
        """Map 0-100 risk score to RiskLevel tier and DecisionAction."""
        pass


class AdvancedHybridRiskEngine(BaseAdvancedHybridRiskEngine):
    """Concrete advanced hybrid risk scoring engine.
    
    Combines:
    1. Calibrated XGBoost Supervised Fraud Probability [0, 1]
    2. Normalized Isolation Forest Anomaly Score [0, 1]
    3. Normalized Behavioral Velocity Score [0, 1]
    
    Scores are aggregated, weighted, bounded to 0.0-100.0, and mapped to configurable decision tiers.
    Includes transparent, evidence-based risk explanations derived strictly from transaction features.
    """

    DEFAULT_WEIGHTS = {
        "xgboost": 0.60,
        "isolation_forest": 0.25,
        "behavioral_velocity": 0.15,
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
        """Ensure component weights are non-negative, finite, and sum to 1.0."""
        required_keys = {"xgboost", "isolation_forest", "behavioral_velocity"}
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
        """Validate input component is a finite float bounded in [0.0, 1.0]."""
        if not isinstance(val, (int, float, np.number)) or not math.isfinite(val):
            raise RiskEngineError(f"Input '{name}' must be a finite float, got {val}")
        
        float_val = float(val)
        if float_val < 0.0 or float_val > 1.0:
            raise RiskEngineError(f"Input '{name}' must be within [0.0, 1.0], got {float_val}")

        return float_val

    def _evaluate_rule_triggers_and_evidence(
        self,
        xgb_prob: float,
        if_score: float,
        vel_score: float,
        context: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Generate rule triggers and evidence explanations derived strictly from real input features."""
        rule_triggers: List[str] = []
        explanations: List[Dict[str, Any]] = []

        # High Supervised Fraud Risk
        if xgb_prob >= 0.70:
            rule_triggers.append("RULE_SUPERVISED_HIGH_PROBABILITY")
            explanations.append({
                "factor": "Supervised Fraud Probability",
                "severity": "HIGH",
                "value": round(xgb_prob, 4),
                "description": f"Calibrated XGBoost model predicted high fraud probability of {xgb_prob:.2%}",
            })

        # High Unsupervised Anomaly Signal
        if if_score >= 0.75:
            rule_triggers.append("RULE_UNSUPERVISED_ANOMALY")
            explanations.append({
                "factor": "Isolation Forest Anomaly Score",
                "severity": "MEDIUM" if if_score < 0.85 else "HIGH",
                "value": round(if_score, 4),
                "description": f"Isolation Forest detected abnormal pattern score of {if_score:.2f}",
            })

        # High Transaction Amount Ratio
        amt_ratio = context.get("amt_to_cust_avg_ratio")
        if amt_ratio is not None and amt_ratio > 3.0:
            rule_triggers.append("RULE_HIGH_AMOUNT_TO_AVG_RATIO")
            explanations.append({
                "factor": "Amount to Average Ratio",
                "severity": "HIGH" if amt_ratio > 5.0 else "MEDIUM",
                "value": round(amt_ratio, 2),
                "description": f"Transaction amount is {amt_ratio:.1f}x higher than historical average",
            })

        # High Transaction Velocity
        tx_count_1h = context.get("cust_tx_count_1h")
        if tx_count_1h is not None and tx_count_1h >= 5.0:
            rule_triggers.append("RULE_HIGH_VELOCITY_1H")
            explanations.append({
                "factor": "1-Hour Transaction Velocity",
                "severity": "HIGH" if tx_count_1h >= 10.0 else "MEDIUM",
                "value": float(tx_count_1h),
                "description": f"Customer initiated {int(tx_count_1h)} transactions in the past hour",
            })

        # New Device / Email Novelty
        if context.get("is_new_device_for_cust") == 1.0:
            rule_triggers.append("RULE_NEW_DEVICE_DETECTED")
            explanations.append({
                "factor": "New Device Channel",
                "severity": "MEDIUM",
                "value": 1.0,
                "description": "Transaction submitted from a previously unseen device channel",
            })

        if context.get("is_new_email_domain") == 1.0:
            rule_triggers.append("RULE_NEW_EMAIL_DOMAIN")
            explanations.append({
                "factor": "New Email Domain",
                "severity": "LOW",
                "value": 1.0,
                "description": "Purchaser email domain has no prior history on this card",
            })

        return rule_triggers, explanations

    def calculate_risk(
        self,
        supervised_xgboost_prob: float,
        isolation_forest_anomaly: float,
        behavioral_velocity: float,
        feature_context: Optional[Dict[str, Any]] = None,
    ) -> AdvancedRiskEngineOutput:
        """Calculate aggregated 0-100 hybrid risk score from component signals."""
        xgb_val = self._validate_input_component("supervised_xgboost_prob", supervised_xgboost_prob)
        if_val = self._validate_input_component("isolation_forest_anomaly", isolation_forest_anomaly)
        vel_val = self._validate_input_component("behavioral_velocity", behavioral_velocity)
        context = feature_context or {}

        raw_weighted_score = (
            self.weights["xgboost"] * xgb_val
            + self.weights["isolation_forest"] * if_val
            + self.weights["behavioral_velocity"] * vel_val
        )

        score_100 = raw_weighted_score * 100.0
        score_bounded = float(np.clip(score_100, 0.0, 100.0))
        risk_score = round(score_bounded, 2)

        risk_level, decision = self.classify_risk_tier(risk_score)

        rule_triggers, explanations = self._evaluate_rule_triggers_and_evidence(
            xgb_val, if_val, vel_val, context
        )

        components = AdvancedModelRiskComponents(
            supervised_xgboost_prob=xgb_val,
            isolation_forest_anomaly_score=if_val,
            behavioral_velocity_score=vel_val,
        )

        logger.debug(
            f"Calculated advanced risk score: {risk_score:.2f} (level={risk_level.value}, decision={decision.value})"
        )

        return AdvancedRiskEngineOutput(
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision,
            components=components,
            rule_triggers=rule_triggers,
            explanations=explanations,
        )

    def classify_risk_tier(self, risk_score: float) -> Tuple[RiskLevel, DecisionAction]:
        """Map a 0-100 risk score to a RiskLevel tier and DecisionAction."""
        if not isinstance(risk_score, (int, float, np.number)) or not math.isfinite(risk_score):
            raise RiskEngineError(f"Risk score must be a finite number, got {risk_score}")

        score = float(risk_score)

        for min_val, max_val, tier, decision in self.risk_bands:
            if min_val <= score <= max_val:
                return tier, decision

        if score < 0.0:
            return RiskLevel.LOW, DecisionAction.APPROVE
        return RiskLevel.CRITICAL, DecisionAction.DECLINE
