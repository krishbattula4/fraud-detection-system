"""Model evaluator abstract contract and concrete implementation for computing classification metrics."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger("model_evaluator")


class BaseModelEvaluator(ABC):
    """Evaluate classifier performance using precision-recall, ROC, and cost-sensitive fraud metrics."""

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Return dictionary of metrics: PR-AUC, ROC-AUC, Precision, Recall, F1, LogLoss."""
        pass


class ModelEvaluator(BaseModelEvaluator):
    """Evaluation suite calculating PR-AUC, ROC-AUC, Precision, Recall, F1, LogLoss, and Confusion Matrix."""

    def evaluate(self, y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Compute evaluation metrics for fraud classification.
        
        PR-AUC is the primary metric for highly imbalanced datasets.
        """
        if y_true is None or y_pred_probs is None or len(y_true) == 0:
            raise ValidationError("Cannot evaluate model on empty ground truth or probabilities.")

        probs_1d = y_pred_probs[:, 1] if y_pred_probs.ndim == 2 else y_pred_probs
        probs_clipped = np.clip(probs_1d, 1e-15, 1 - 1e-15)

        # Binary label predictions derived from threshold
        y_pred = (probs_clipped >= threshold).astype(int)

        # Primary imbalance metric: PR-AUC
        try:
            pr_auc = float(average_precision_score(y_true, probs_clipped))
        except Exception:
            pr_auc = 0.0

        try:
            roc_auc = float(roc_auc_score(y_true, probs_clipped))
        except Exception:
            roc_auc = 0.5

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            loss = float(log_loss(y_true, probs_clipped))
        except Exception:
            loss = 0.0

        # Confusion matrix breakdown
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        metrics = {
            "pr_auc": round(pr_auc, 5),
            "roc_auc": round(roc_auc, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1_score": round(f1, 5),
            "log_loss": round(loss, 5),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": round(fpr, 5),
            "false_negative_rate": round(fnr, 5),
            "eval_threshold": float(threshold),
        }

        logger.info(
            f"Evaluation result (threshold={threshold}): PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}."
        )

        return metrics
