"""Advanced ML models wrapper and registry for version 2.0.0 artifacts."""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from src.core.config import get_settings
from src.core.exceptions import ModelNotFoundError, InferenceError
from src.core.logging import get_logger

logger = get_logger("advanced_models")

def get_advanced_artifact_dir() -> str:
    """Return storage directory for advanced model artifacts."""
    settings = get_settings()
    base_dir = os.path.dirname(settings.model_dir)
    adv_dir = os.path.join(base_dir, "advanced_artifacts")
    os.makedirs(adv_dir, exist_ok=True)
    return adv_dir

class AdvancedXGBoostModel:
    """Supervised XGBoost model trained on multi-entity transaction features."""

    def __init__(self, model_version: str = "2.0.0"):
        self.model_version = model_version
        self.model: Optional[XGBClassifier] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, scale_pos_weight: float = 25.0) -> None:
        """Fit XGBoost classifier on training split."""
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return estimated fraud probability array in [0.0, 1.0]."""
        if self.model is None:
            raise InferenceError("Advanced XGBoost model is not loaded or fitted.")
        return self.model.predict_proba(X)[:, 1]


class AdvancedIsolationForestModel:
    """Unsupervised Isolation Forest anomaly detector for advanced transaction vectors."""

    def __init__(self, model_version: str = "2.0.0"):
        self.model_version = model_version
        self.model: Optional[IsolationForest] = None

    def fit(self, X_train: pd.DataFrame, contamination: float = 0.03) -> None:
        """Fit Isolation Forest on training vectors."""
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train)

    def score_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly score array (higher score = more anomalous)."""
        if self.model is None:
            raise InferenceError("Advanced Isolation Forest model is not loaded or fitted.")
        # Invert score_samples so higher is more anomalous
        raw_scores = -self.model.score_samples(X)
        return raw_scores


class AdvancedModelRegistry:
    """Artifact loader and registry for v2.0.0 advanced models."""

    @staticmethod
    def save_artifact(obj: Any, filename: str, metadata: Dict[str, Any]) -> str:
        """Save model object and JSON metadata to advanced_artifacts directory."""
        adv_dir = get_advanced_artifact_dir()
        file_path = os.path.join(adv_dir, filename)
        joblib.dump(obj, file_path)
        
        meta_path = os.path.join(adv_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved advanced artifact '{filename}' to {adv_dir}.")
        return file_path

    @staticmethod
    def load_artifact(filename: str) -> Any:
        """Load model object from advanced_artifacts directory."""
        adv_dir = get_advanced_artifact_dir()
        file_path = os.path.join(adv_dir, filename)
        if not os.path.exists(file_path):
            raise ModelNotFoundError(f"Advanced model artifact '{filename}' not found at {file_path}")
        return joblib.load(file_path)
