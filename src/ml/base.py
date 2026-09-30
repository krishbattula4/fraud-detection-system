"""Abstract base classes for ML models, metadata tracking, and artifact managers."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import numpy as np
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata record for model registry tracking."""
    model_name: str = Field(..., description="Unique model identifier")
    version: str = Field(..., description="Semantic version tag")
    algorithm: str = Field(..., description="Algorithm name e.g. XGBoost, IsolationForest")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameter dict")
    feature_columns: List[str] = Field(default_factory=list, description="Ordered feature column names")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Evaluation metrics dictionary")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Creation timestamp")


class BaseFraudModel(ABC):
    """Base contract for any model operating within the fraud system."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name identifier of the model."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Model artifact version string."""
        pass


class BaseAnomalyModel(BaseFraudModel):
    """Base contract for unsupervised anomaly detection models."""

    @abstractmethod
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Output raw uncalibrated anomaly scores for input feature array.
        
        CONVENTION: Higher output score indicates higher likelihood of being an anomaly.
        """
        pass


class ModelArtifactManager(ABC):
    """Interface for model serialization, versioning, and disk persistence."""

    @abstractmethod
    def save_artifact(self, model_object: Any, artifact_name: str, version: str, metadata: Optional[ModelMetadata] = None) -> str:
        """Serialize model artifact to configured storage path."""
        pass

    @abstractmethod
    def load_artifact(self, artifact_name: str, version: Optional[str] = None) -> Any:
        """Load serialized model artifact from storage path."""
        pass
