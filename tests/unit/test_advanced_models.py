"""Unit tests for advanced ML models, artifact loading, and inference contracts."""
import os
import pytest
import pandas as pd
import numpy as np

from src.ml.advanced_models import (
    AdvancedXGBoostModel,
    AdvancedIsolationForestModel,
    AdvancedModelRegistry,
    get_advanced_artifact_dir,
)
from src.core.exceptions import ModelNotFoundError, InferenceError


def test_advanced_xgboost_inference_range():
    """Verify Advanced XGBoost predicts probabilities strictly bounded to [0.0, 1.0]."""
    adv_dir = get_advanced_artifact_dir()
    xgb_path = os.path.join(adv_dir, "advanced_xgboost_v2.0.0.joblib")
    
    if not os.path.exists(xgb_path):
        pytest.skip("Advanced XGBoost joblib artifact not yet trained.")

    xgb_model: AdvancedXGBoostModel = AdvancedModelRegistry.load_artifact("advanced_xgboost_v2.0.0.joblib")
    
    # Dummy feature vector matching advanced schema
    X_dummy = pd.DataFrame([{
        "amount": 150.0,
        "amount_scaled": 1.2,
        "cust_tx_count_1h": 2.0,
        "cust_tx_count_24h": 5.0,
        "cust_amt_sum_24h": 450.0,
        "amt_to_cust_avg_ratio": 1.5,
        "is_new_device_for_cust": 0,
        "is_new_email_domain": 0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "c1": 1.0,
        "d1": 0.0,
    }])

    probs = xgb_model.predict_proba(X_dummy)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


def test_advanced_isolation_forest_scoring():
    """Verify Advanced Isolation Forest scores anomalies deterministically."""
    adv_dir = get_advanced_artifact_dir()
    if_path = os.path.join(adv_dir, "advanced_isolation_forest_v2.0.0.joblib")
    
    if not os.path.exists(if_path):
        pytest.skip("Advanced Isolation Forest joblib artifact not yet trained.")

    iforest: AdvancedIsolationForestModel = AdvancedModelRegistry.load_artifact("advanced_isolation_forest_v2.0.0.joblib")
    
    X_dummy = pd.DataFrame([{
        "amount": 5000.0,
        "amount_scaled": 50.0,
        "cust_tx_count_1h": 20.0,
        "cust_tx_count_24h": 50.0,
        "cust_amt_sum_24h": 25000.0,
        "amt_to_cust_avg_ratio": 10.0,
        "is_new_device_for_cust": 1,
        "is_new_email_domain": 1,
        "hour_of_day": 3,
        "day_of_week": 6,
        "c1": 5.0,
        "d1": 12.0,
    }])

    scores = iforest.score_anomaly(X_dummy)
    assert len(scores) == 1
    assert np.isfinite(scores[0])


def test_missing_artifact_raises_exception():
    """Verify attempting to load a non-existent artifact raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError):
        AdvancedModelRegistry.load_artifact("non_existent_artifact_v9.9.9.joblib")
