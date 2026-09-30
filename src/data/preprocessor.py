"""Data preprocessor abstract contract and concrete implementation for leakage-safe feature scaling."""
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
from src.core.exceptions import ValidationError, ModelNotFoundError
from src.core.logging import get_logger

logger = get_logger("data_preprocessor")

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class BaseDataPreprocessor(ABC):
    """Abstract interface for feature scaling and transformations.
    
    CRITICAL LEAKAGE REQUIREMENT:
    The fit() method must ONLY be called on training split data.
    transform() is applied independently to validation, test, or single inference payloads.
    """

    @abstractmethod
    def fit(self, X_train: pd.DataFrame) -> "BaseDataPreprocessor":
        """Fit feature scalers/transformers strictly on training split."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations to feature set."""
        pass

    @abstractmethod
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training data and transform in a single step."""
        pass


class DataPreprocessor(BaseDataPreprocessor):
    """Concrete data preprocessor enforcing leakage-safe scaling.
    
    Fits RobustScaler on 'Amount' and 'Time' features strictly on the training set.
    PCA features V1..V28 are preserved without re-scaling.
    """

    def __init__(self):
        self.time_scaler = RobustScaler()
        self.amount_scaler = RobustScaler()
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "DataPreprocessor":
        """Fit feature scalers strictly on training split."""
        if X_train is None or X_train.empty:
            raise ValidationError("Cannot fit preprocessor on empty or None DataFrame.")

        missing_features = [col for col in FEATURE_COLUMNS if col not in X_train.columns]
        if missing_features:
            raise ValidationError(f"Training features missing required columns: {missing_features}")

        logger.info("Fitting RobustScalers on 'Time' and 'Amount' training features...")
        self.time_scaler.fit(X_train[["Time"]].values)
        self.amount_scaler.fit(X_train[["Amount"]].values)
        self.is_fitted = True
        logger.info("Preprocessor fitting completed successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted scaling transformations to feature matrix.
        
        CRITICAL: Uses fitted parameters from fit() without refitting.
        """
        if not self.is_fitted:
            raise ValidationError("Preprocessor must be fitted on training data before calling transform().")

        if X is None or X.empty:
            raise ValidationError("Cannot transform empty or None DataFrame.")

        missing_features = [col for col in FEATURE_COLUMNS if col not in X.columns]
        if missing_features:
            raise ValidationError(f"Input features missing required columns: {missing_features}")

        # Scale Time and Amount using fitted state
        scaled_time = self.time_scaler.transform(X[["Time"]].values)
        scaled_amount = self.amount_scaler.transform(X[["Amount"]].values)

        # Retain PCA features V1..V28 as-is
        pca_features = X[[f"V{i}" for i in range(1, 29)]].values

        # Concatenate in order [Scaled_Time, V1..V28, Scaled_Amount]
        transformed_matrix = np.hstack([scaled_time, pca_features, scaled_amount])
        return transformed_matrix

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed feature matrix."""
        self.fit(X_train)
        return self.transform(X_train)

    def save_state(self, filepath: str) -> str:
        """Serialize fitted preprocessor state to disk artifact."""
        if not self.is_fitted:
            raise ValidationError("Cannot save un-fitted preprocessor state.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "time_scaler": self.time_scaler,
            "amount_scaler": self.amount_scaler,
            "is_fitted": self.is_fitted,
            "feature_columns": FEATURE_COLUMNS,
        }
        joblib.dump(state, filepath)
        logger.info(f"Preprocessor state saved to '{filepath}'.")
        return filepath

    def load_state(self, filepath: str) -> "DataPreprocessor":
        """Deserialize preprocessor state from disk artifact."""
        if not os.path.exists(filepath):
            raise ModelNotFoundError(f"Preprocessor artifact file not found at '{filepath}'.")
        state = joblib.load(filepath)
        self.time_scaler = state["time_scaler"]
        self.amount_scaler = state["amount_scaler"]
        self.is_fitted = state["is_fitted"]
        logger.info(f"Preprocessor state successfully loaded from '{filepath}'.")
        return self
