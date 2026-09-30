"""Unit tests for Advanced Hybrid Risk Engine, score aggregation, tier classification, and rule triggers."""
import pytest
import math
from src.engine.advanced_risk_engine import AdvancedHybridRiskEngine
from src.schemas.prediction import RiskLevel, DecisionAction
from src.core.exceptions import RiskEngineError


def test_advanced_risk_engine_valid_calculation():
    """Verify hybrid risk engine aggregates scores correctly bounded between 0.0 and 100.0."""
    engine = AdvancedHybridRiskEngine()

    output = engine.calculate_risk(
        supervised_xgboost_prob=0.80,
        isolation_forest_anomaly=0.60,
        behavioral_velocity=0.40,
        feature_context={"amt_to_cust_avg_ratio": 4.5, "cust_tx_count_1h": 6.0},
    )

    # Weighted calculation: 0.60*0.80 + 0.25*0.60 + 0.15*0.40 = 0.48 + 0.15 + 0.06 = 0.69 -> 69.0
    assert math.isclose(output.risk_score, 69.0, abs_tol=0.1)
    assert output.risk_level == RiskLevel.MEDIUM
    assert output.decision == DecisionAction.FLAG_FOR_REVIEW
    assert "RULE_SUPERVISED_HIGH_PROBABILITY" in output.rule_triggers
    assert "RULE_HIGH_AMOUNT_TO_AVG_RATIO" in output.rule_triggers
    assert "RULE_HIGH_VELOCITY_1H" in output.rule_triggers


def test_advanced_risk_engine_critical_tier():
    """Verify high component scores yield CRITICAL risk level and DECLINE decision action."""
    engine = AdvancedHybridRiskEngine()

    output = engine.calculate_risk(
        supervised_xgboost_prob=0.95,
        isolation_forest_anomaly=0.90,
        behavioral_velocity=0.85,
    )

    assert output.risk_score >= 90.0
    assert output.risk_level == RiskLevel.CRITICAL
    assert output.decision == DecisionAction.DECLINE


def test_advanced_risk_engine_low_tier():
    """Verify low component scores yield LOW risk level and APPROVE decision action."""
    engine = AdvancedHybridRiskEngine()

    output = engine.calculate_risk(
        supervised_xgboost_prob=0.05,
        isolation_forest_anomaly=0.10,
        behavioral_velocity=0.0,
    )

    assert output.risk_score < 40.0
    assert output.risk_level == RiskLevel.LOW
    assert output.decision == DecisionAction.APPROVE


def test_advanced_risk_engine_weight_validation():
    """Verify invalid weights raise RiskEngineError."""
    with pytest.raises(RiskEngineError):
        AdvancedHybridRiskEngine(weights={"xgboost": 0.50, "isolation_forest": 0.30})  # Missing behavioral_velocity

    with pytest.raises(RiskEngineError):
        AdvancedHybridRiskEngine(weights={"xgboost": 0.50, "isolation_forest": 0.30, "behavioral_velocity": 0.30})  # Sums to 1.10
