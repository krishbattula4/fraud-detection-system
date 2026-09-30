"""Probability calibrator and anomaly score normalizer implementations."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.core.exceptions import ValidationError, InferenceError
from src.core.logging import get_logger

logger = get_logger("ml_calibration")


class BaseScoreCalibrator(ABC):
    """Calibrate uncalibrated supervised probabilities."""

    @abstractmethod
    def fit(self, uncalibrated_probs: np.ndarray, y_val: np.ndarray) -> "BaseScoreCalibrator":
        pass

    @abstractmethod
    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        pass


class BaseScoreNormalizer(ABC):
    """Normalize raw anomaly scores into bounded [0, 1] range."""

    @abstractmethod
    def fit(self, raw_scores: np.ndarray) -> "BaseScoreNormalizer":
        pass

    @abstractmethod
    def normalize(self, raw_scores: np.ndarray) -> np.ndarray:
        pass


class ProbabilityCalibrator(BaseScoreCalibrator):
    """Platt scaling probability calibrator using Logistic Regression on validation split."""

    def __init__(self):
        self.calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.is_fitted: bool = False

    def fit(self, uncalibrated_probs: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        if uncalibrated_probs is None or len(uncalibrated_probs) == 0:
            raise ValidationError("Cannot fit ProbabilityCalibrator on empty probabilities.")
        
        # Reshape to 2D array if 1D array of fraud probabilities
        probs_2d = uncalibrated_probs.reshape(-1, 1) if uncalibrated_probs.ndim == 1 else uncalibrated_probs[:, [1]]
        
        logger.info(f"Fitting Platt ProbabilityCalibrator on {len(probs_2d)} validation samples...")
        self.calibrator.fit(probs_2d, y_val)
        self.is_fitted = True
        return self

    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """Transform uncalibrated fraud probabilities to well-calibrated probabilities in [0, 1]."""
        if not self.is_fitted:
            raise InferenceError("ProbabilityCalibrator must be fitted before calibrate().")
        
        probs_2d = uncalibrated_probs.reshape(-1, 1) if uncalibrated_probs.ndim == 1 else uncalibrated_probs[:, [1]]
        calibrated_probs = self.calibrator.predict_proba(probs_2d)[:, 1]
        return np.clip(calibrated_probs, 0.0, 1.0)


class MinMaxScoreNormalizer(BaseScoreNormalizer):
    """Min-Max scaler mapping continuous anomaly scores (higher = more anomalous) to bounded [0.0, 1.0]."""

    def __init__(self):
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.is_fitted: bool = False

    def fit(self, raw_scores: np.ndarray) -> "MinMaxScoreNormalizer":
        if raw_scores is None or len(raw_scores) == 0:
            raise ValidationError("Cannot fit MinMaxScoreNormalizer on empty raw scores.")

        scores_flat = raw_scores.ravel()
        self.min_val = float(np.min(scores_flat))
        self.max_val = float(np.max(scores_flat))

        # Handle zero range corner case
        if self.max_val == self.min_val:
            self.max_val += 1e-6

        logger.info(f"Fitting MinMaxScoreNormalizer (min={self.min_val:.4f}, max={self.max_val:.4f})...")
        self.is_fitted = True
        return self

    def normalize(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to [0.0, 1.0] component scores."""
        if not self.is_fitted:
            raise InferenceError("MinMaxScoreNormalizer must be fitted before normalize().")

        scores_flat = raw_scores.ravel()
        normalized = (scores_flat - self.min_val) / (self.max_val - self.min_val)
        return np.clip(normalized, 0.0, 1.0)
