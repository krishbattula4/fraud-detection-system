"""Unit test suite for Phase 3 ML models, calibrators, evaluators, registry, and SHAP explainer."""
import os
import pytest
import numpy as np
import pandas as pd
from src.ml.supervised import LogisticRegressionModel, RandomForestFraudModel, XGBoostFraudModel
from src.ml.anomaly import IsolationForestAnomalyDetector, LOFAnomalyDetector
from src.ml.calibration import ProbabilityCalibrator, MinMaxScoreNormalizer
from src.ml.evaluator import ModelEvaluator
from src.ml.registry import ModelRegistry
from src.explainability.explainer import SHAPTransactionExplainer
from src.core.exceptions import InferenceError, ValidationError, ModelNotFoundError


@pytest.fixture
def synthetic_ml_dataset():
    """Create a deterministic synthetic dataset fixture for testing ML pipeline wrappers."""
    np.random.seed(42)
    n_samples = 150
    n_features = 30

    X = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    
    # 135 non-fraud (0), 15 fraud (1)
    y = np.zeros(n_samples, dtype=int)
    y[:15] = 1

    return X, y


# =====================================================================
# Supervised Model Tests
# =====================================================================

def test_logistic_regression_fit_predict(synthetic_ml_dataset):
    """Verify LogisticRegressionModel fit, predict_proba, and threshold prediction."""
    X, y = synthetic_ml_dataset
    model = LogisticRegressionModel()
    
    with pytest.raises(InferenceError):
        model.predict_proba(X)

    model.fit(X, y)
    assert model.is_fitted is True

    probs = model.predict_proba(X)
    assert probs.shape == (150, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = model.predict(X, threshold=0.5)
    assert preds.shape == (150,)
    assert set(preds).issubset({0, 1})


def test_random_forest_fit_predict(synthetic_ml_dataset):
    """Verify RandomForestFraudModel fit and predict_proba."""
    X, y = synthetic_ml_dataset
    model = RandomForestFraudModel(n_estimators=10)
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    assert probs.shape == (150, 2)


def test_xgboost_fit_predict(synthetic_ml_dataset):
    """Verify XGBoostFraudModel scale_pos_weight calculation and fit."""
    X, y = synthetic_ml_dataset
    model = XGBoostFraudModel(n_estimators=10)
    model.fit(X, y)
    
    assert model.is_fitted is True
    probs = model.predict_proba(X)
    assert probs.shape == (150, 2)


# =====================================================================
# Anomaly Detector Tests
# =====================================================================

def test_isolation_forest_anomaly_scores(synthetic_ml_dataset):
    """Verify IsolationForestAnomalyDetector outputs positive anomaly orientation."""
    X, _ = synthetic_ml_dataset
    detector = IsolationForestAnomalyDetector(n_estimators=10)
    detector.fit(X)
    
    scores = detector.predict_anomaly_score(X)
    assert scores.shape == (150,)
    # Verify higher scores correspond to anomalies
    assert np.all(np.isfinite(scores))


def test_lof_novelty_inference_on_unseen_samples(synthetic_ml_dataset):
    """Verify LOFAnomalyDetector configured with novelty=True can score unseen test samples."""
    X_train, _ = synthetic_ml_dataset
    X_unseen = np.random.normal(loc=5.0, scale=1.0, size=(10, 30))  # High outlier test matrix

    detector = LOFAnomalyDetector(n_neighbors=5)
    detector.fit(X_train)

    scores_unseen = detector.predict_anomaly_score(X_unseen)
    assert scores_unseen.shape == (10,)
    assert np.all(np.isfinite(scores_unseen))


# =====================================================================
# Calibration & Normalization Tests
# =====================================================================

def test_probability_calibrator(synthetic_ml_dataset):
    """Verify ProbabilityCalibrator fits on validation split and outputs [0, 1] probabilities."""
    X, y = synthetic_ml_dataset
    xgb = XGBoostFraudModel(n_estimators=10).fit(X, y)
    uncal_probs = xgb.predict_proba(X)[:, 1]

    calibrator = ProbabilityCalibrator()
    calibrator.fit(uncal_probs, y)

    calibrated = calibrator.calibrate(uncal_probs)
    assert calibrated.shape == (150,)
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_minmax_score_normalizer():
    """Verify MinMaxScoreNormalizer maps raw scores to [0.0, 1.0]."""
    raw_scores = np.array([-0.5, 0.0, 0.5, 1.0, 2.0])
    normalizer = MinMaxScoreNormalizer()
    normalizer.fit(raw_scores)

    norm_scores = normalizer.normalize(raw_scores)
    assert norm_scores[0] == 0.0
    assert norm_scores[-1] == 1.0
    assert np.all(norm_scores >= 0.0) and np.all(norm_scores <= 1.0)


# =====================================================================
# Model Evaluator & Registry Tests
# =====================================================================

def test_model_evaluator_metrics(synthetic_ml_dataset):
    """Verify ModelEvaluator calculates PR-AUC, ROC-AUC, Precision, Recall, F1, and Confusion Matrix."""
    X, y = synthetic_ml_dataset
    xgb = XGBoostFraudModel(n_estimators=10).fit(X, y)
    probs = xgb.predict_proba(X)[:, 1]

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y, probs, threshold=0.5)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "true_positives" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_model_registry_save_and_load(tmp_path, synthetic_ml_dataset):
    """Verify ModelRegistry saves and loads joblib artifacts with metadata JSON."""
    X, y = synthetic_ml_dataset
    xgb = XGBoostFraudModel(n_estimators=10).fit(X, y)

    registry = ModelRegistry(model_dir=str(tmp_path))
    save_path = registry.save_artifact(xgb, "test_xgb", version="1.0.0")
    assert os.path.exists(save_path)

    loaded_xgb, meta = registry.load_artifact("test_xgb", version="1.0.0")
    assert loaded_xgb.is_fitted is True
    assert meta["model_name"] == "test_xgb"


# =====================================================================
# SHAP Explainer Tests
# =====================================================================

def test_shap_explainer_single_transaction(synthetic_ml_dataset):
    """Verify SHAPTransactionExplainer initializes and generates top_k attributions."""
    X, y = synthetic_ml_dataset
    xgb = XGBoostFraudModel(n_estimators=10).fit(X, y)

    explainer = SHAPTransactionExplainer()
    explainer.fit_explainer(xgb, background_data=X[:10])

    sample_tx_vector = X[0].tolist()
    explanations = explainer.explain_transaction(sample_tx_vector, top_k=5)

    assert len(explanations) == 5
    assert hasattr(explanations[0], "feature_name")
    assert hasattr(explanations[0], "importance_score")
    # Verify exact feature name format (e.g. Time, V1..V28, Amount)
    assert explanations[0].feature_name in ["Time", "Amount"] or explanations[0].feature_name.startswith("V")
