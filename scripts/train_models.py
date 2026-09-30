"""Dedicated model training, validation calibration, hybrid tuning, test evaluation, and artifact serialization script."""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.ml.supervised import LogisticRegressionModel, RandomForestFraudModel, XGBoostFraudModel
from src.ml.anomaly import IsolationForestAnomalyDetector, LOFAnomalyDetector
from src.ml.calibration import ProbabilityCalibrator, MinMaxScoreNormalizer
from src.ml.evaluator import ModelEvaluator
from src.ml.registry import ModelRegistry
from src.engine.risk_engine import HybridRiskEngine
from src.explainability.explainer import SHAPTransactionExplainer
from src.core.logging import get_logger

logger = get_logger("train_models_script")


def run_training_pipeline():
    start_time = time.time()
    logger.info("Starting Phase 6B Real Model Training Pipeline...")

    # 1. Load Real Dataset
    loader = DataLoader()
    raw_df = loader.load_raw_data("creditcard.csv")
    total_rows, total_cols = raw_df.shape
    logger.info(f"Loaded raw dataset: {total_rows} rows, {total_cols} columns.")

    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df = loader.split_chronological(raw_df, train_ratio=0.70, val_ratio=0.15)
    
    y_train = train_df["Class"].values
    y_val = val_df["Class"].values
    y_test = test_df["Class"].values

    train_fraud = int(np.sum(y_train == 1))
    val_fraud = int(np.sum(y_val == 1))
    test_fraud = int(np.sum(y_test == 1))

    logger.info(
        f"Splits summary:\n"
        f"  Train: {len(train_df)} rows ({train_fraud} fraud)\n"
        f"  Val:   {len(val_df)} rows ({val_fraud} fraud)\n"
        f"  Test:  {len(test_df)} rows ({test_fraud} fraud)"
    )

    # 3. Fit DataPreprocessor ONLY on Train Split
    logger.info("Fitting DataPreprocessor strictly on Train split...")
    preprocessor = DataPreprocessor().fit(train_df)
    
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    # 4. Train Supervised Baseline Models
    logger.info("Training 1/5: Logistic Regression Baseline...")
    t0 = time.time()
    lr_model = LogisticRegressionModel(version="1.0.0").fit(X_train, y_train)
    lr_time = time.time() - t0

    logger.info("Training 2/5: Random Forest Baseline...")
    t0 = time.time()
    rf_model = RandomForestFraudModel(version="1.0.0", n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_time = time.time() - t0

    # 5. Train XGBoost Primary Supervised Model with Dynamic scale_pos_weight
    n_neg_train = np.sum(y_train == 0)
    n_pos_train = np.sum(y_train == 1)
    spw_exact = float(n_neg_train / n_pos_train)
    logger.info(f"Training 3/5: XGBoost (scale_pos_weight={spw_exact:.4f})...")
    t0 = time.time()
    xgb_model = XGBoostFraudModel(
        version="1.0.0", n_estimators=100, max_depth=6, learning_rate=0.05, scale_pos_weight=spw_exact, random_state=42
    ).fit(X_train, y_train)
    xgb_time = time.time() - t0

    # 6. Train Unsupervised Anomaly Detectors
    logger.info("Training 4/5: Isolation Forest Anomaly Detector...")
    t0 = time.time()
    iforest_model = IsolationForestAnomalyDetector(version="1.0.0", n_estimators=100, random_state=42).fit(X_train)
    iforest_time = time.time() - t0

    logger.info("Training 5/5: Local Outlier Factor (LOF) Anomaly Detector (novelty=True)...")
    t0 = time.time()
    lof_model = LOFAnomalyDetector(version="1.0.0", n_neighbors=20).fit(X_train)
    lof_time = time.time() - t0
    logger.info(f"LOF fitting completed in {lof_time:.2f} seconds.")

    # 7. Fit Calibrator and Score Normalizers strictly on VALIDATION Split
    logger.info("Fitting ProbabilityCalibrator strictly on Validation split...")
    uncal_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    calibrator = ProbabilityCalibrator().fit(uncal_val_probs, y_val)

    logger.info("Fitting MinMaxScoreNormalizers strictly on Validation split...")
    val_raw_if = iforest_model.predict_anomaly_score(X_val)
    norm_if = MinMaxScoreNormalizer().fit(val_raw_if)

    val_raw_lof = lof_model.predict_anomaly_score(X_val)
    norm_lof = MinMaxScoreNormalizer().fit(val_raw_lof)

    # 8. Deterministic Candidate Hybrid Weight Search on VALIDATION ONLY
    logger.info("Evaluating hybrid weight candidates on Validation split...")
    evaluator = ModelEvaluator()
    
    val_cal_xgb_probs = calibrator.calibrate(uncal_val_probs)
    val_norm_if = norm_if.normalize(val_raw_if)
    val_norm_lof = norm_lof.normalize(val_raw_lof)

    candidate_weights = [
        {"xgboost": 0.60, "isolation_forest": 0.25, "lof": 0.15},
        {"xgboost": 0.70, "isolation_forest": 0.20, "lof": 0.10},
        {"xgboost": 0.80, "isolation_forest": 0.10, "lof": 0.10},
        {"xgboost": 0.50, "isolation_forest": 0.30, "lof": 0.20},
    ]

    best_weights = candidate_weights[0]
    best_val_prauc = -1.0

    for w_dict in candidate_weights:
        engine = HybridRiskEngine(weights=w_dict)
        val_risk_scores = np.array([
            engine.calculate_risk(p, i, l).risk_score / 100.0
            for p, i, l in zip(val_cal_xgb_probs, val_norm_if, val_norm_lof)
        ])
        val_metrics = evaluator.evaluate(y_val, val_risk_scores)
        pr_auc = val_metrics["pr_auc"]
        logger.info(f"Candidate weights {w_dict} -> Val PR-AUC = {pr_auc:.4f}")
        if pr_auc > best_val_prauc:
            best_val_prauc = pr_auc
            best_weights = w_dict

    logger.info(f"Selected Best Hybrid Weights from Validation: {best_weights} (Val PR-AUC = {best_val_prauc:.4f})")
    final_risk_engine = HybridRiskEngine(weights=best_weights)

    # 9. Freeze Decisions & Perform Final Evaluation on untouched TEST Split
    logger.info("Performing Final Model Evaluation on untouched Test split...")

    # Supervised predictions on Test
    test_lr_probs = lr_model.predict_proba(X_test)[:, 1]
    test_rf_probs = rf_model.predict_proba(X_test)[:, 1]
    test_uncal_xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    test_cal_xgb_probs = calibrator.calibrate(test_uncal_xgb_probs)

    # Anomaly predictions on Test
    test_raw_if = iforest_model.predict_anomaly_score(X_test)
    test_norm_if = norm_if.normalize(test_raw_if)

    test_raw_lof = lof_model.predict_anomaly_score(X_test)
    test_norm_lof = norm_lof.normalize(test_raw_lof)

    # Hybrid Risk output on Test
    test_hybrid_outputs = [
        final_risk_engine.calculate_risk(p, i, l)
        for p, i, l in zip(test_cal_xgb_probs, test_norm_if, test_norm_lof)
    ]
    test_hybrid_scores = np.array([out.risk_score / 100.0 for out in test_hybrid_outputs])

    # Calculate metrics for all models on Test split
    metrics_summary = {
        "dataset": {
            "total_rows": total_rows,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_fraud": train_fraud,
            "val_fraud": val_fraud,
            "test_fraud": test_fraud,
            "scale_pos_weight": spw_exact,
        },
        "runtimes_sec": {
            "logistic_regression": round(lr_time, 2),
            "random_forest": round(rf_time, 2),
            "xgboost": round(xgb_time, 2),
            "isolation_forest": round(iforest_time, 2),
            "lof": round(lof_time, 2),
        },
        "hybrid_configuration": {
            "weights": best_weights,
            "validation_pr_auc": round(best_val_prauc, 4),
        },
        "test_evaluations": {
            "logistic_regression": evaluator.evaluate(y_test, test_lr_probs),
            "random_forest": evaluator.evaluate(y_test, test_rf_probs),
            "xgboost_uncalibrated": evaluator.evaluate(y_test, test_uncal_xgb_probs),
            "xgboost_calibrated": evaluator.evaluate(y_test, test_cal_xgb_probs),
            "isolation_forest": evaluator.evaluate(y_test, test_norm_if),
            "lof": evaluator.evaluate(y_test, test_norm_lof),
            "hybrid_risk_engine": evaluator.evaluate(y_test, test_hybrid_scores),
        },
    }

    logger.info("--- FINAL TEST EVALUATION RESULTS ---")
    for m_name, m_val in metrics_summary["test_evaluations"].items():
        logger.info(
            f"  {m_name:25s} | PR-AUC: {m_val['pr_auc']:.4f} | ROC-AUC: {m_val['roc_auc']:.4f} | "
            f"Precision: {m_val['precision']:.4f} | Recall: {m_val['recall']:.4f} | F1: {m_val['f1_score']:.4f}"
        )

    # 10. Serialize Real Model Artifacts using ModelRegistry
    logger.info("Serializing real model artifacts to ModelRegistry...")
    registry = ModelRegistry()

    registry.save_artifact(preprocessor, "preprocessor", version="1.0.0")
    registry.save_artifact(lr_model, "logistic_regression", version="1.0.0")
    registry.save_artifact(rf_model, "random_forest", version="1.0.0")
    registry.save_artifact(xgb_model, "xgboost", version="1.0.0")
    registry.save_artifact(iforest_model, "isolation_forest", version="1.0.0")
    registry.save_artifact(lof_model, "local_outlier_factor", version="1.0.0")
    registry.save_artifact(calibrator, "probability_calibrator", version="1.0.0")
    registry.save_artifact(norm_if, "iforest_normalizer", version="1.0.0")
    registry.save_artifact(norm_lof, "lof_normalizer", version="1.0.0")

    # Fit and save SHAP explainer on XGBoost
    explainer = SHAPTransactionExplainer().fit_explainer(xgb_model, background_data=X_train[:100])
    registry.save_artifact(explainer, "shap_explainer", version="1.0.0")

    # Save summary metrics JSON file
    metrics_path = os.path.join(registry.model_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved complete evaluation metrics JSON to '{metrics_path}'.")

    total_time = time.time() - start_time
    logger.info(f"Phase 6B Training Pipeline Completed Successfully in {total_time:.2f} seconds!")
    return metrics_summary


if __name__ == "__main__":
    run_training_pipeline()
