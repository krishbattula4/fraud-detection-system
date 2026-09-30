"""Machine learning interfaces, supervised/anomaly models, calibrators, evaluators, and registry."""
from src.ml.base import BaseFraudModel, BaseAnomalyModel, ModelArtifactManager, ModelMetadata
from src.ml.supervised import (
    BaseSupervisedFraudModel,
    LogisticRegressionModel,
    RandomForestFraudModel,
    XGBoostFraudModel,
)
from src.ml.anomaly import (
    BaseAnomalyDetector,
    IsolationForestAnomalyDetector,
    LOFAnomalyDetector,
)
from src.ml.calibration import (
    BaseScoreCalibrator,
    BaseScoreNormalizer,
    ProbabilityCalibrator,
    MinMaxScoreNormalizer,
)
from src.ml.evaluator import BaseModelEvaluator, ModelEvaluator
from src.ml.registry import ModelRegistry

__all__ = [
    "BaseFraudModel",
    "BaseAnomalyModel",
    "ModelArtifactManager",
    "ModelMetadata",
    "BaseSupervisedFraudModel",
    "LogisticRegressionModel",
    "RandomForestFraudModel",
    "XGBoostFraudModel",
    "BaseAnomalyDetector",
    "IsolationForestAnomalyDetector",
    "LOFAnomalyDetector",
    "BaseScoreCalibrator",
    "BaseScoreNormalizer",
    "ProbabilityCalibrator",
    "MinMaxScoreNormalizer",
    "BaseModelEvaluator",
    "ModelEvaluator",
    "ModelRegistry",
]
