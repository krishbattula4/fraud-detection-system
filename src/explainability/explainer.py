"""SHAP transaction explainer abstract contract and concrete implementation."""
from abc import ABC, abstractmethod
from typing import List, Any, Optional
import numpy as np
import shap
from src.schemas.prediction import FeatureImportanceExplanation
from src.core.exceptions import ValidationError, InferenceError
from src.core.logging import get_logger

logger = get_logger("shap_explainer")

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class BaseTransactionExplainer(ABC):
    """Abstract interface for calculating SHAP-based feature attributions for transactions."""

    @abstractmethod
    def fit_explainer(self, model: Any, background_data: Optional[np.ndarray] = None) -> "BaseTransactionExplainer":
        """Initialize SHAP explainer on trained model artifact and background sample matrix."""
        pass

    @abstractmethod
    def explain_transaction(
        self, feature_vector: List[float], top_k: int = 5
    ) -> List[FeatureImportanceExplanation]:
        """Compute local feature attributions for a single transaction vector."""
        pass


class SHAPTransactionExplainer(BaseTransactionExplainer):
    """Calculates per-transaction local feature attributions using SHAP TreeExplainer."""

    def __init__(self):
        self.explainer: Optional[Any] = None
        self.is_fitted: bool = False

    def fit_explainer(self, model: Any, background_data: Optional[np.ndarray] = None) -> "SHAPTransactionExplainer":
        """Initialize SHAP TreeExplainer on trained tree model (e.g. XGBoost)."""
        if model is None:
            raise ValidationError("Cannot fit SHAP explainer on None model object.")

        # Extract underlying XGBClassifier if wrapped in XGBoostFraudModel
        model_to_explain = getattr(model, "model", model)

        try:
            logger.info("Initializing SHAP TreeExplainer...")
            try:
                self.explainer = shap.TreeExplainer(model_to_explain)
            except Exception:
                if background_data is not None and len(background_data) > 0:
                    self.explainer = shap.TreeExplainer(model_to_explain, data=background_data)
                else:
                    self.explainer = shap.Explainer(model_to_explain)
            self.is_fitted = True
            logger.info("SHAP TreeExplainer successfully initialized.")
            return self
        except Exception as e:
            logger.error(f"Failed to initialize SHAP TreeExplainer: {str(e)}")
            raise ValidationError(f"SHAP TreeExplainer initialization failed: {str(e)}")

    def explain_transaction(
        self, feature_vector: List[float], top_k: int = 5
    ) -> List[FeatureImportanceExplanation]:
        """Compute local feature attributions for a single transaction vector.
        
        retains exact dataset feature names (e.g. V14, Amount, Time) without inventing unverified human descriptions.
        """
        if not self.is_fitted or self.explainer is None:
            raise InferenceError("SHAPTransactionExplainer must be initialized via fit_explainer() before explain_transaction().")

        if feature_vector is None or len(feature_vector) != len(FEATURE_NAMES):
            raise ValidationError(
                f"Feature vector length ({len(feature_vector) if feature_vector else 0}) must equal {len(FEATURE_NAMES)}."
            )

        X_single = np.array(feature_vector).reshape(1, -1)

        try:
            shap_values = self.explainer.shap_values(X_single)
            
            # Extract 1D array of SHAP values for single sample
            if isinstance(shap_values, list):
                # Binary classification list [non-fraud, fraud]
                vals = shap_values[1][0]
            elif shap_values.ndim == 2:
                vals = shap_values[0]
            else:
                vals = shap_values.ravel()

            explanations = []
            for name, val_feat, impact in zip(FEATURE_NAMES, feature_vector, vals):
                explanations.append(
                    FeatureImportanceExplanation(
                        feature_name=name,
                        feature_value=float(val_feat),
                        importance_score=float(impact),
                    )
                )

            # Sort by absolute SHAP impact magnitude descending
            explanations.sort(key=lambda x: abs(x.importance_score), reverse=True)
            return explanations[:top_k]

        except Exception as e:
            logger.error(f"Failed to compute SHAP values for transaction: {str(e)}")
            raise InferenceError(f"SHAP calculation failed: {str(e)}")
