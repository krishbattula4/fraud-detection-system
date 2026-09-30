"""Concrete supervised model wrappers (Logistic Regression, Random Forest, XGBoost)."""
from typing import Optional, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.ml.base import BaseFraudModel
from src.core.exceptions import ValidationError, InferenceError
from src.core.logging import get_logger

logger = get_logger("supervised_models")


class BaseSupervisedFraudModel(BaseFraudModel):
    """Abstract interface for supervised fraud classification models."""

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

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels using specified decision threshold."""
        probs = self.predict_proba(X)
        fraud_probs = probs[:, 1] if probs.ndim == 2 else probs
        return (fraud_probs >= threshold).astype(int)


class LogisticRegressionModel(BaseSupervisedFraudModel):
    """Logistic Regression baseline classifier."""

    def __init__(self, version: str = "1.0.0", C: float = 1.0, random_state: int = 42):
        super().__init__(name="logistic_regression", version=version)
        self.model = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionModel":
        if X_train is None or len(X_train) == 0:
            raise ValidationError("Cannot fit Logistic Regression on empty training data.")
        logger.info(f"Fitting LogisticRegressionModel on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise InferenceError("LogisticRegressionModel must be fitted before predict_proba().")
        return self.model.predict_proba(X)


class RandomForestFraudModel(BaseSupervisedFraudModel):
    """Random Forest baseline tree classifier."""

    def __init__(self, version: str = "1.0.0", n_estimators: int = 100, max_depth: Optional[int] = 10, random_state: int = 42):
        super().__init__(name="random_forest", version=version)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestFraudModel":
        if X_train is None or len(X_train) == 0:
            raise ValidationError("Cannot fit RandomForestFraudModel on empty training data.")
        logger.info(f"Fitting RandomForestFraudModel on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise InferenceError("RandomForestFraudModel must be fitted before predict_proba().")
        return self.model.predict_proba(X)


class XGBoostFraudModel(BaseSupervisedFraudModel):
    """Primary supervised XGBoost classifier."""

    def __init__(
        self,
        version: str = "1.0.0",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42
    ):
        super().__init__(name="xgboost", version=version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model: Optional[XGBClassifier] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostFraudModel":
        if X_train is None or len(X_train) == 0:
            raise ValidationError("Cannot fit XGBoostFraudModel on empty training data.")

        # Calculate scale_pos_weight if not explicitly provided
        spw = self.scale_pos_weight
        if spw is None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            spw = float(n_neg / n_pos) if n_pos > 0 else 1.0

        logger.info(f"Fitting XGBoostFraudModel (scale_pos_weight={spw:.2f}) on {len(X_train)} samples...")
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise InferenceError("XGBoostFraudModel must be fitted before predict_proba().")
        return self.model.predict_proba(X)
