"""Transaction prediction service layer integrating feature preprocessing, model inference, hybrid risk calculation, SHAP explainability, and database persistence."""
import uuid
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set
from sqlalchemy.orm import Session

from src.schemas.transaction import TransactionInput
from src.schemas.prediction import (
    RiskPredictionResponse,
    RiskLevel,
    DecisionAction,
    ModelRiskComponents,
    FeatureImportanceExplanation,
)
from src.schemas.alert import AlertStatus
from src.engine.risk_engine import HybridRiskEngine, RiskEngineOutput
from src.ml.registry import ModelRegistry
from src.data.preprocessor import DataPreprocessor, FEATURE_COLUMNS
from src.explainability.explainer import SHAPTransactionExplainer
from src.db.models import TransactionRecord, PredictionRecord, AlertRecord
from src.db.repository import TransactionRepository, PredictionRepository, AlertRepository
from src.core.exceptions import ModelNotFoundError, InferenceError, ValidationError, DatabaseError
from src.core.logging import get_logger

logger = get_logger("prediction_service")


class PredictionService:
    """Orchestrates end-to-end transaction fraud risk evaluation and persistence."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        model_version: str = "1.0.0",
        risk_engine: Optional[HybridRiskEngine] = None,
        preprocessor: Optional[Any] = None,
        xgb_model: Optional[Any] = None,
        iforest_model: Optional[Any] = None,
        lof_model: Optional[Any] = None,
        calibrator: Optional[Any] = None,
        iforest_normalizer: Optional[Any] = None,
        lof_normalizer: Optional[Any] = None,
        explainer: Optional[Any] = None,
        alert_levels: Optional[Set[RiskLevel]] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.model_version = model_version
        self.risk_engine = risk_engine or HybridRiskEngine()
        
        # Injectable components (used directly or loaded from registry)
        self.preprocessor = preprocessor
        self.xgb_model = xgb_model
        self.iforest_model = iforest_model
        self.lof_model = lof_model
        self.calibrator = calibrator
        self.iforest_normalizer = iforest_normalizer
        self.lof_normalizer = lof_normalizer
        self.explainer = explainer

        # Configurable alert creation threshold
        self.alert_levels = alert_levels or {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def _ensure_artifacts_loaded(self) -> None:
        """Load required Phase 3 model artifacts from ModelRegistry if not explicitly injected."""
        if (
            self.preprocessor is not None
            and self.xgb_model is not None
            and self.iforest_model is not None
            and self.lof_model is not None
            and self.calibrator is not None
            and self.iforest_normalizer is not None
            and self.lof_normalizer is not None
        ):
            return  # All components already injected

        logger.info(f"Loading model artifacts (version={self.model_version}) from registry...")

        try:
            if self.xgb_model is None:
                self.xgb_model, _ = self.registry.load_artifact("xgboost", version=self.model_version)
            if self.iforest_model is None:
                self.iforest_model, _ = self.registry.load_artifact("isolation_forest", version=self.model_version)
            if self.lof_model is None:
                self.lof_model, _ = self.registry.load_artifact("local_outlier_factor", version=self.model_version)
            if self.calibrator is None:
                self.calibrator, _ = self.registry.load_artifact("probability_calibrator", version=self.model_version)
            if self.iforest_normalizer is None:
                self.iforest_normalizer, _ = self.registry.load_artifact("iforest_normalizer", version=self.model_version)
            if self.lof_normalizer is None:
                self.lof_normalizer, _ = self.registry.load_artifact("lof_normalizer", version=self.model_version)
            if self.preprocessor is None:
                self.preprocessor, _ = self.registry.load_artifact("preprocessor", version=self.model_version)
        except Exception as e:
            logger.error(f"Failed to load required model artifacts: {str(e)}")
            raise ModelNotFoundError(
                f"Required model artifacts for version '{self.model_version}' are unavailable: {str(e)}"
            )

    def predict_transaction_risk(
        self,
        transaction_input: TransactionInput,
        db: Optional[Session] = None,
    ) -> RiskPredictionResponse:
        """Execute complete inference pipeline for a single transaction.
        
        1. Ensure model artifacts exist (fail explicitly if missing)
        2. Format input to DataFrame & apply leakage-safe preprocessor
        3. Run model inferences (XGBoost, Isolation Forest, LOF)
        4. Apply calibrator & score normalizers
        5. Pass signals to Hybrid Risk Engine
        6. Compute SHAP explanations (if explainer initialized)
        7. Persist transaction, prediction, and alert to database (if DB session provided)
        8. Return structured prediction response
        """
        self._ensure_artifacts_loaded()

        # Step 1: Extract feature vector & convert to DataFrame matching FEATURE_COLUMNS
        feature_vector = transaction_input.to_feature_vector()
        raw_dict = transaction_input.model_dump()
        
        # Construct DataFrame using ordered feature vector [Time, V1..V28, Amount]
        df_single = pd.DataFrame([feature_vector], columns=FEATURE_COLUMNS)

        # Step 2: Transform features using preprocessor
        try:
            transformed_matrix = self.preprocessor.transform(df_single)
        except Exception as e:
            logger.error(f"Preprocessor transformation failed: {str(e)}")
            raise InferenceError(f"Preprocessing transformation failed: {str(e)}")

        # Step 3: Run Model Inferences
        try:
            # XGBoost probability
            uncal_prob = self.xgb_model.predict_proba(transformed_matrix)
            raw_xgb_prob = float(uncal_prob[:, 1][0] if uncal_prob.ndim == 2 else uncal_prob[0])
            xgb_calibrated_prob = float(self.calibrator.calibrate(np.array([raw_xgb_prob]))[0])

            # Isolation Forest anomaly score
            raw_if_score = self.iforest_model.predict_anomaly_score(transformed_matrix)
            if_normalized = float(self.iforest_normalizer.normalize(raw_if_score)[0])

            # LOF anomaly score
            raw_lof_score = self.lof_model.predict_anomaly_score(transformed_matrix)
            lof_normalized = float(self.lof_normalizer.normalize(raw_lof_score)[0])

        except Exception as e:
            logger.error(f"Model inference or calibration step failed: {str(e)}")
            raise InferenceError(f"Model inference failed: {str(e)}")

        # Step 4: Hybrid Risk Aggregation
        risk_output: RiskEngineOutput = self.risk_engine.calculate_risk(
            xgboost_prob=xgb_calibrated_prob,
            iforest_anomaly=if_normalized,
            lof_anomaly=lof_normalized,
        )

        # Step 5: Explainability (SHAP attributions)
        explanations: List[FeatureImportanceExplanation] = []
        if self.explainer is not None:
            try:
                explanations = self.explainer.explain_transaction(feature_vector, top_k=5)
            except Exception as e:
                logger.warning(f"SHAP explanation failed, returning empty explanations: {str(e)}")

        # Step 6: Generate IDs & Timestamp
        prediction_id = f"pred-{uuid.uuid4().hex[:12]}"
        transaction_db_id = f"tx-{uuid.uuid4().hex[:12]}"
        evaluated_at_iso = datetime.now(timezone.utc).isoformat()
        now_dt = datetime.now(timezone.utc)

        # Step 7: Persistence (if DB session provided)
        if db is not None:
            tx_repo = TransactionRepository(db)
            pred_repo = PredictionRepository(db)
            alert_repo = AlertRepository(db)

            # 7a. Persist Transaction
            tx_record = TransactionRecord(
                id=transaction_db_id,
                transaction_id=transaction_input.transaction_id,
                time=transaction_input.time,
                amount=transaction_input.amount,
                raw_features=raw_dict,
                created_at=now_dt,
            )
            tx_repo.add(tx_record)

            # 7b. Persist Prediction
            explanations_dict = [exp.model_dump() for exp in explanations] if explanations else []
            pred_record = PredictionRecord(
                id=prediction_id,
                transaction_record_id=transaction_db_id,
                risk_score=risk_output.risk_score,
                risk_level=risk_output.risk_level.value,
                decision=risk_output.decision.value,
                xgboost_prob=xgb_calibrated_prob,
                iforest_anomaly=if_normalized,
                lof_anomaly=lof_normalized,
                explanations=explanations_dict,
                model_version=self.model_version,
                evaluated_at=now_dt,
            )
            pred_repo.add(pred_record)

            # 7c. Persist Alert if risk level triggers alert condition
            if risk_output.risk_level in self.alert_levels:
                alert_id = f"alt-{uuid.uuid4().hex[:12]}"
                alert_record = AlertRecord(
                    id=alert_id,
                    prediction_id=prediction_id,
                    risk_score=risk_output.risk_score,
                    risk_level=risk_output.risk_level.value,
                    status=AlertStatus.OPEN.value,
                    created_at=now_dt,
                )
                alert_repo.add(alert_record)

        return RiskPredictionResponse(
            prediction_id=prediction_id,
            transaction_id=transaction_input.transaction_id,
            risk_score=risk_output.risk_score,
            risk_level=risk_output.risk_level,
            decision=risk_output.decision,
            model_components=risk_output.components,
            explanations=explanations,
            model_version=self.model_version,
            evaluated_at=evaluated_at_iso,
        )
