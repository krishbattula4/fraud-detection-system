"""Advanced prediction service layer integrating IEEE-CIS feature preprocessing, model inference, advanced hybrid risk calculation, and database persistence."""
import uuid
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session

from src.schemas.advanced_transaction import (
    AdvancedTransactionInput,
    AdvancedRiskPredictionResponse,
    AdvancedModelRiskComponents,
)
from src.schemas.prediction import RiskLevel, DecisionAction
from src.schemas.alert import AlertStatus
from src.engine.advanced_risk_engine import AdvancedHybridRiskEngine, AdvancedRiskEngineOutput
from src.ml.advanced_models import (
    AdvancedXGBoostModel,
    AdvancedIsolationForestModel,
    AdvancedModelRegistry,
    get_advanced_artifact_dir,
)
from src.db.models import TransactionRecord, PredictionRecord, AlertRecord
from src.db.repository import TransactionRepository, PredictionRepository, AlertRepository
from src.core.exceptions import ModelNotFoundError, InferenceError, DatabaseError
from src.core.logging import get_logger

logger = get_logger("advanced_prediction_service")


class AdvancedPredictionService:
    """Orchestrates end-to-end advanced risk evaluation using authentic IEEE-CIS model artifacts."""

    ADVANCED_FEATURE_COLUMNS = [
        "amount",
        "amount_scaled",
        "cust_tx_count_1h",
        "cust_tx_count_24h",
        "cust_amt_sum_24h",
        "amt_to_cust_avg_ratio",
        "is_new_device_for_cust",
        "is_new_email_domain",
        "hour_of_day",
        "day_of_week",
        "c1",
        "d1",
    ]

    def __init__(
        self,
        model_version: str = "2.0.0-authentic-ieee",
        risk_engine: Optional[AdvancedHybridRiskEngine] = None,
        xgb_model: Optional[Any] = None,
        iforest_model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        calibrator: Optional[Any] = None,
        anomaly_scaler: Optional[Any] = None,
    ):
        self.model_version = model_version
        self.risk_engine = risk_engine or AdvancedHybridRiskEngine()

        self.xgb_model = xgb_model
        self.iforest_model = iforest_model
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self.anomaly_scaler = anomaly_scaler

    def _ensure_artifacts_loaded(self) -> None:
        """Load required Phase 7 authentic IEEE-CIS joblib artifacts if not explicitly injected."""
        if (
            self.xgb_model is not None
            and self.iforest_model is not None
            and self.preprocessor is not None
            and self.calibrator is not None
            and self.anomaly_scaler is not None
        ):
            return

        logger.info(f"Loading advanced model artifacts (version={self.model_version})...")

        try:
            adv_dir = get_advanced_artifact_dir()
            
            if self.xgb_model is None:
                self.xgb_model = AdvancedModelRegistry.load_artifact("advanced_xgboost_v2.0.0.joblib")
            if self.iforest_model is None:
                self.iforest_model = AdvancedModelRegistry.load_artifact("advanced_isolation_forest_v2.0.0.joblib")
            if self.preprocessor is None:
                self.preprocessor = AdvancedModelRegistry.load_artifact("advanced_preprocessor_v2.0.0.joblib")
            if self.calibrator is None:
                self.calibrator = AdvancedModelRegistry.load_artifact("advanced_calibrator_v2.0.0.joblib")
            if self.anomaly_scaler is None:
                self.anomaly_scaler = AdvancedModelRegistry.load_artifact("advanced_anomaly_scaler_v2.0.0.joblib")
        except Exception as e:
            logger.error(f"Failed to load required advanced model artifacts: {str(e)}")
            raise ModelNotFoundError(
                f"Required advanced model artifacts for version '{self.model_version}' are unavailable: {str(e)}"
            )

    def extract_feature_vector(
        self, transaction_input: AdvancedTransactionInput
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process AdvancedTransactionInput into 12-feature DataFrame matching model schema."""
        behavioral_dict = self.preprocessor.compute_behavioral_features(transaction_input)

        raw_amt = float(transaction_input.amount)
        if self.preprocessor.is_fitted:
            amt_scaled = float(self.preprocessor.scaler.transform([[raw_amt]])[0][0])
        else:
            amt_scaled = raw_amt / 100.0

        c1 = 0.0
        if transaction_input.counting_features:
            c1 = float(transaction_input.counting_features.get("C1", transaction_input.counting_features.get("c1", 0.0)))

        d1 = 0.0
        if transaction_input.timedelta_features:
            d1 = float(transaction_input.timedelta_features.get("D1", transaction_input.timedelta_features.get("d1", 0.0)))

        feature_dict = {
            "amount": raw_amt,
            "amount_scaled": amt_scaled,
            "cust_tx_count_1h": behavioral_dict["cust_tx_count_1h"],
            "cust_tx_count_24h": behavioral_dict["cust_tx_count_24h"],
            "cust_amt_sum_24h": behavioral_dict["cust_amt_sum_24h"],
            "amt_to_cust_avg_ratio": behavioral_dict["amt_to_cust_avg_ratio"],
            "is_new_device_for_cust": behavioral_dict["is_new_device_for_cust"],
            "is_new_email_domain": behavioral_dict["is_new_email_domain"],
            "hour_of_day": behavioral_dict["hour_of_day"],
            "day_of_week": behavioral_dict["day_of_week"],
            "c1": c1,
            "d1": d1,
        }

        df_single = pd.DataFrame([feature_dict], columns=self.ADVANCED_FEATURE_COLUMNS)
        return df_single, feature_dict

    def calculate_velocity_score(self, feature_dict: Dict[str, Any]) -> float:
        """Calculate normalized behavioral velocity score [0.0, 1.0] from count & ratio signals."""
        c_1h = feature_dict.get("cust_tx_count_1h", 0.0)
        c_24h = feature_dict.get("cust_tx_count_24h", 0.0)
        amt_ratio = feature_dict.get("amt_to_cust_avg_ratio", 1.0)

        # Normalize velocity components safely
        v_1h = min(c_1h / 10.0, 1.0)
        v_24h = min(c_24h / 30.0, 1.0)
        v_ratio = min((amt_ratio - 1.0) / 9.0, 1.0) if amt_ratio > 1.0 else 0.0

        raw_vel = 0.50 * v_1h + 0.30 * v_ratio + 0.20 * v_24h
        return float(np.clip(raw_vel, 0.0, 1.0))

    def predict_advanced_risk(
        self,
        transaction_input: AdvancedTransactionInput,
        db: Optional[Session] = None,
    ) -> AdvancedRiskPredictionResponse:
        """Execute complete advanced inference pipeline for a multi-entity transaction."""
        self._ensure_artifacts_loaded()

        # Step 1: Feature Extraction
        df_features, feature_dict = self.extract_feature_vector(transaction_input)

        # Step 2: Model Inferences
        try:
            raw_xgb_prob = self.xgb_model.predict_proba(df_features)[0]
            calibrated_xgb_prob = float(self.calibrator.predict(np.array([raw_xgb_prob]))[0])
            calibrated_xgb_prob = float(np.clip(calibrated_xgb_prob, 0.0, 1.0))

            raw_if_score = self.iforest_model.score_anomaly(df_features)[0]
            scaled_if_score = float(self.anomaly_scaler.transform([[raw_if_score]])[0][0])
            scaled_if_score = float(np.clip(scaled_if_score, 0.0, 1.0))

            velocity_score = self.calculate_velocity_score(feature_dict)

        except Exception as e:
            logger.error(f"Advanced model inference failed: {str(e)}")
            raise InferenceError(f"Advanced model inference failed: {str(e)}")

        # Step 3: Hybrid Risk Aggregation
        risk_output: AdvancedRiskEngineOutput = self.risk_engine.calculate_risk(
            supervised_xgboost_prob=calibrated_xgb_prob,
            isolation_forest_anomaly=scaled_if_score,
            behavioral_velocity=velocity_score,
            feature_context=feature_dict,
        )

        # Step 4: System Timestamps & Identifiers
        prediction_id = f"adv-pred-{uuid.uuid4().hex[:12]}"
        transaction_db_id = f"adv-tx-{uuid.uuid4().hex[:12]}"
        evaluated_at_iso = datetime.now(timezone.utc).isoformat()
        now_dt = datetime.now(timezone.utc)

        # Step 5: Persistence (if DB session provided)
        if db is not None:
            tx_repo = TransactionRepository(db)
            pred_repo = PredictionRepository(db)
            alert_repo = AlertRepository(db)

            tx_record = TransactionRecord(
                id=transaction_db_id,
                transaction_id=transaction_input.transaction_id,
                time=transaction_input.timestamp,
                amount=transaction_input.amount,
                raw_features=transaction_input.model_dump(),
                created_at=now_dt,
            )
            tx_repo.add(tx_record)

            pred_record = PredictionRecord(
                id=prediction_id,
                transaction_record_id=transaction_db_id,
                risk_score=risk_output.risk_score,
                risk_level=risk_output.risk_level.value,
                decision=risk_output.decision.value,
                xgboost_prob=calibrated_xgb_prob,
                iforest_anomaly=scaled_if_score,
                lof_anomaly=velocity_score,  # Store velocity score in anomaly column for advanced pipeline
                explanations=risk_output.explanations,
                model_version=self.model_version,
                evaluated_at=now_dt,
            )
            pred_repo.add(pred_record)

            if risk_output.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
                alert_id = f"adv-alt-{uuid.uuid4().hex[:12]}"
                alert_record = AlertRecord(
                    id=alert_id,
                    prediction_id=prediction_id,
                    risk_score=risk_output.risk_score,
                    risk_level=risk_output.risk_level.value,
                    status=AlertStatus.OPEN.value,
                    created_at=now_dt,
                )
                alert_repo.add(alert_record)

        return AdvancedRiskPredictionResponse(
            prediction_id=prediction_id,
            transaction_id=transaction_input.transaction_id,
            risk_score=risk_output.risk_score,
            risk_level=risk_output.risk_level,
            decision=risk_output.decision,
            model_components=risk_output.components,
            behavioral_signals=feature_dict,
            rule_triggers=risk_output.rule_triggers,
            explanations=risk_output.explanations,
            model_version=self.model_version,
            evaluated_at=evaluated_at_iso,
        )
