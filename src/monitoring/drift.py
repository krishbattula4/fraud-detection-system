"""Abstract drift detector interface contract."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from src.schemas.metrics import DriftReport


class BaseDriftDetector(ABC):
    """Abstract interface for monitoring distribution drift in feature inputs and output risk scores."""

    @abstractmethod
    def set_reference_baseline(self, baseline_features: np.ndarray, baseline_scores: np.ndarray) -> None:
        """Register baseline distributions (e.g. from validation dataset split)."""
        pass

    @abstractmethod
    def calculate_drift(self, current_features: np.ndarray, current_scores: np.ndarray) -> DriftReport:
        """Compute KS-statistic or Population Stability Index (PSI) drift report."""
        pass
