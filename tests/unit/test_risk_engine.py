"""Unit test suite for Phase 4 HybridRiskEngine logic."""
import pytest
import math
import numpy as np

from src.engine.risk_engine import HybridRiskEngine, RiskEngineOutput
from src.schemas.prediction import RiskLevel, DecisionAction
from src.core.exceptions import RiskEngineError


@pytest.fixture
def risk_engine():
    """Default HybridRiskEngine instance."""
    return HybridRiskEngine()


def test_valid_hybrid_score(risk_engine):
    """1. Test calculate_risk with valid component inputs."""
    result = risk_engine.calculate_risk(
        xgboost_prob=0.8,
        iforest_anomaly=0.4,
        lof_anomaly=0.2,
    )
    # Expected weighted sum: 0.6*0.8 + 0.25*0.4 + 0.15*0.2 = 0.48 + 0.10 + 0.03 = 0.61 -> score = 61.0
    assert isinstance(result, RiskEngineOutput)
    assert result.risk_score == 61.0
    assert result.risk_level == RiskLevel.MEDIUM
    assert result.decision == DecisionAction.FLAG_FOR_REVIEW
    assert result.components.xgboost_calibrated_prob == 0.8
    assert result.components.isolation_forest_anomaly_score == 0.4
    assert result.components.lof_anomaly_score == 0.2


def test_probability_bounds(risk_engine):
    """2. Test probability bounds checking (reject < 0.0 or > 1.0)."""
    with pytest.raises(RiskEngineError, match="xgboost_prob"):
        risk_engine.calculate_risk(-0.1, 0.5, 0.5)

    with pytest.raises(RiskEngineError, match="xgboost_prob"):
        risk_engine.calculate_risk(1.05, 0.5, 0.5)


def test_anomaly_score_bounds(risk_engine):
    """3. Test anomaly score bounds checking."""
    with pytest.raises(RiskEngineError, match="iforest_anomaly"):
        risk_engine.calculate_risk(0.5, -0.01, 0.5)

    with pytest.raises(RiskEngineError, match="lof_anomaly"):
        risk_engine.calculate_risk(0.5, 0.5, 1.2)


def test_score_always_within_0_100(risk_engine):
    """4. Test that for any extreme valid inputs, output score is strictly bounded [0.0, 100.0]."""
    res_min = risk_engine.calculate_risk(0.0, 0.0, 0.0)
    assert res_min.risk_score == 0.0
    assert res_min.risk_level == RiskLevel.LOW

    res_max = risk_engine.calculate_risk(1.0, 1.0, 1.0)
    assert res_max.risk_score == 100.0
    assert res_max.risk_level == RiskLevel.CRITICAL

    # Test grid of inputs
    for prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for if_score in [0.0, 0.5, 1.0]:
            for lof_score in [0.0, 0.5, 1.0]:
                out = risk_engine.calculate_risk(prob, if_score, lof_score)
                assert 0.0 <= out.risk_score <= 100.0


def test_risk_band_classification(risk_engine):
    """5. Test risk band classification boundaries (LOW, MEDIUM, HIGH, CRITICAL)."""
    # 0-39 LOW
    low_res = risk_engine.calculate_risk(0.1, 0.1, 0.1)
    assert low_res.risk_level == RiskLevel.LOW

    # 40-69 MEDIUM
    med_res = risk_engine.calculate_risk(0.5, 0.5, 0.5)
    assert med_res.risk_level == RiskLevel.MEDIUM

    # 70-89 HIGH
    high_res = risk_engine.calculate_risk(0.8, 0.8, 0.8)
    assert high_res.risk_level == RiskLevel.HIGH

    # 90-100 CRITICAL
    crit_res = risk_engine.calculate_risk(0.95, 0.95, 0.95)
    assert crit_res.risk_level == RiskLevel.CRITICAL


def test_decision_mapping(risk_engine):
    """6. Test decision mapping (APPROVE, FLAG_FOR_REVIEW, DECLINE)."""
    assert risk_engine.classify_risk_tier(20.0) == (RiskLevel.LOW, DecisionAction.APPROVE)
    assert risk_engine.classify_risk_tier(50.0) == (RiskLevel.MEDIUM, DecisionAction.FLAG_FOR_REVIEW)
    assert risk_engine.classify_risk_tier(80.0) == (RiskLevel.HIGH, DecisionAction.FLAG_FOR_REVIEW)
    assert risk_engine.classify_risk_tier(95.0) == (RiskLevel.CRITICAL, DecisionAction.DECLINE)


def test_invalid_weights():
    """7. Test invalid weights rejection (sum != 1.0 or negative)."""
    with pytest.raises(RiskEngineError, match="weights must sum to 1.0"):
        HybridRiskEngine(weights={"xgboost": 0.5, "isolation_forest": 0.5, "lof": 0.5})

    with pytest.raises(RiskEngineError, match="cannot be negative"):
        HybridRiskEngine(weights={"xgboost": 1.2, "isolation_forest": -0.2, "lof": 0.0})


def test_nan_inf_rejection(risk_engine):
    """8. Test NaN/Inf input rejection."""
    with pytest.raises(RiskEngineError, match="must be a finite float"):
        risk_engine.calculate_risk(float("nan"), 0.5, 0.5)

    with pytest.raises(RiskEngineError, match="must be a finite float"):
        risk_engine.calculate_risk(0.5, float("inf"), 0.5)


def test_configurable_threshold_behavior():
    """9. Test custom weights and custom risk bands."""
    custom_weights = {"xgboost": 0.80, "isolation_forest": 0.10, "lof": 0.10}
    custom_engine = HybridRiskEngine(weights=custom_weights)

    out = custom_engine.calculate_risk(1.0, 0.0, 0.0)
    assert out.risk_score == 80.0
    assert out.risk_level == RiskLevel.HIGH


def test_deterministic_hybrid_calculation(risk_engine):
    """10. Test that identical inputs yield identical outputs deterministically."""
    out1 = risk_engine.calculate_risk(0.732, 0.415, 0.889)
    out2 = risk_engine.calculate_risk(0.732, 0.415, 0.889)
    assert out1.risk_score == out2.risk_score
    assert out1.risk_level == out2.risk_level
    assert out1.decision == out2.decision
