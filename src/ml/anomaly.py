"""Concrete anomaly detector wrappers (Isolation Forest, Local Outlier Factor)."""
from typing import Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.ml.base import BaseAnomalyModel
from src.core.exceptions import ValidationError, InferenceError
from src.core.logging import get_logger

logger = get_logger("anomaly_models")


class BaseAnomalyDetector(BaseAnomalyModel):
    """Abstract interface for unsupervised anomaly detectors."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self.is_fitted: bool = False

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version


class IsolationForestAnomalyDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector wrapper."""

    def __init__(self, version: str = "1.0.0", n_estimators: int = 100, contamination: str = "auto", random_state: int = 42):
        super().__init__(name="isolation_forest", version=version)
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray) -> "IsolationForestAnomalyDetector":
        if X_train is None or len(X_train) == 0:
            raise ValidationError("Cannot fit IsolationForestAnomalyDetector on empty training data.")
        logger.info(f"Fitting IsolationForestAnomalyDetector on {len(X_train)} samples...")
        self.model.fit(X_train)
        self.is_fitted = True
        return self

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Calculate continuous anomaly score (higher score = more anomalous).
        
        scikit-learn score_samples() outputs negative values (more negative = more anomalous).
        We invert sign so higher output = higher anomaly likelihood.
        """
        if not self.is_fitted:
            raise InferenceError("IsolationForestAnomalyDetector must be fitted before predict_anomaly_score().")
        raw_scores = self.model.score_samples(X)
        return -raw_scores


class LOFAnomalyDetector(BaseAnomalyDetector):
    """Local Outlier Factor anomaly detector wrapper configured with novelty=True for unseen inference."""

    def __init__(self, version: str = "1.0.0", n_neighbors: int = 20, contamination: str = "auto"):
        super().__init__(name="local_outlier_factor", version=version)
        # CRITICAL: novelty=True allows predicting/scoring on new unseen transaction samples!
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray) -> "LOFAnomalyDetector":
        if X_train is None or len(X_train) == 0:
            raise ValidationError("Cannot fit LOFAnomalyDetector on empty training data.")
        logger.info(f"Fitting LOFAnomalyDetector (novelty=True) on {len(X_train)} samples...")
        self.model.fit(X_train)
        self.is_fitted = True
        return self

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Calculate continuous LOF anomaly score for unseen samples (higher score = more anomalous).
        
        Inverts scikit-learn score_samples() output sign.
        """
        if not self.is_fitted:
            raise InferenceError("LOFAnomalyDetector must be fitted before predict_anomaly_score().")
        raw_scores = self.model.score_samples(X)
        return -raw_scores
