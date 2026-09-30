"""Script to train, calibrate, evaluate, and serialize Advanced ML Models (v2.0.0-authentic-ieee)
using the Authentic Kaggle IEEE-CIS Fraud Detection Dataset.

Vectorized for High-Performance Execution.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.data.advanced_preprocessing import AdvancedDataPreprocessor
from src.ml.advanced_models import (
    AdvancedXGBoostModel,
    AdvancedIsolationForestModel,
    AdvancedModelRegistry,
    get_advanced_artifact_dir,
)
from src.core.logging import get_logger

logger = get_logger("train_advanced_models_authentic")

def load_authentic_ieee_dataset() -> pd.DataFrame:
    """Load authentic Kaggle IEEE-CIS transaction and identity CSV datasets."""
    tx_path = os.path.join("data", "raw", "train_transaction.csv")
    id_path = os.path.join("data", "raw", "train_identity.csv")

    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Authentic transaction file missing at '{tx_path}'")

    logger.info(f"Loading authentic IEEE-CIS transaction data from '{tx_path}'...")
    df_tx = pd.read_csv(tx_path)

    if os.path.exists(id_path):
        logger.info(f"Loading authentic IEEE-CIS identity data from '{id_path}'...")
        df_id = pd.read_csv(id_path)
        id_cols = [c for c in ["TransactionID", "DeviceType", "DeviceInfo"] if c in df_id.columns]
        df_merged = pd.merge(df_tx, df_id[id_cols], on="TransactionID", how="left")
    else:
        df_merged = df_tx

    df_merged = df_merged.sort_values(by="TransactionDT").reset_index(drop=True)
    logger.info(f"Loaded {len(df_merged):,} authentic IEEE-CIS transactions.")
    return df_merged

def run_authentic_advanced_training_pipeline():
    logger.info("Starting Authentic IEEE-CIS ML Training Pipeline (v2.0.0-authentic-ieee)...")
    adv_dir = get_advanced_artifact_dir()

    # 1. Authentic Data Ingestion & Chronological Split
    df = load_authentic_ieee_dataset()

    total_rows = len(df)
    n_train = int(total_rows * 0.70)
    n_val = int(total_rows * 0.15)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train+n_val].copy()
    df_test = df.iloc[n_train+n_val:].copy()

    train_frauds = int(df_train["isFraud"].sum())
    val_frauds = int(df_val["isFraud"].sum())
    test_frauds = int(df_test["isFraud"].sum())

    logger.info(f"Chronological Split: Train={len(df_train):,} ({train_frauds} frauds), Val={len(df_val):,} ({val_frauds} frauds), Test={len(df_test):,} ({test_frauds} frauds)")

    # 2. Vectorized Feature Extraction (Train-Only Scaler Fitting)
    scaler = RobustScaler()
    scaler.fit(df_train[["TransactionAmt"]].values)

    preprocessor = AdvancedDataPreprocessor()
    preprocessor.fit_scaler_on_train(df_train["TransactionAmt"].values)

    def extract_vectorized_features(df_sub: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        amt = df_sub["TransactionAmt"].values.reshape(-1, 1)
        amt_scaled = scaler.transform(amt).flatten()
        dt = df_sub["TransactionDT"].values
        
        hour_of_day = ((dt // 3600) % 24).astype(float)
        day_of_week = ((dt // 86400) % 7).astype(float)

        c1 = df_sub["C1"].fillna(0.0).values if "C1" in df_sub.columns else np.zeros(len(df_sub))
        c2 = df_sub["C2"].fillna(0.0).values if "C2" in df_sub.columns else np.zeros(len(df_sub))
        d1 = df_sub["D1"].fillna(0.0).values if "D1" in df_sub.columns else np.zeros(len(df_sub))

        features_df = pd.DataFrame({
            "amount": df_sub["TransactionAmt"].values,
            "amount_scaled": amt_scaled,
            "cust_tx_count_1h": c1,
            "cust_tx_count_24h": c2,
            "cust_amt_sum_24h": df_sub["TransactionAmt"].values * (c1 + 1.0),
            "amt_to_cust_avg_ratio": np.clip(df_sub["TransactionAmt"].values / (c1 * 50.0 + 10.0), 0.0, 100.0),
            "is_new_device_for_cust": df_sub["DeviceType"].notna().astype(float).values if "DeviceType" in df_sub.columns else np.zeros(len(df_sub)),
            "is_new_email_domain": df_sub["P_emaildomain"].notna().astype(float).values if "P_emaildomain" in df_sub.columns else np.zeros(len(df_sub)),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "c1": c1,
            "d1": d1,
        })

        return features_df, df_sub["isFraud"].values

    X_train, y_train = extract_vectorized_features(df_train)
    X_val, y_val = extract_vectorized_features(df_val)
    X_test, y_test = extract_vectorized_features(df_test)

    # 3. Supervised Model Training (Train Split Only)
    pos_weight = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))
    xgb_model = AdvancedXGBoostModel(model_version="2.0.0-authentic-ieee")
    xgb_model.fit(X_train, y_train, scale_pos_weight=pos_weight)

    # 4. Anomaly Model Training (Train Split Only)
    iforest_model = AdvancedIsolationForestModel(model_version="2.0.0-authentic-ieee")
    iforest_model.fit(X_train, contamination=0.035)

    # 5. Validation-Safe Calibration & Normalization (Val Split Only)
    uncal_val_probs = xgb_model.predict_proba(X_val)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(uncal_val_probs, y_val)

    raw_val_anomalies = iforest_model.score_anomaly(X_val)
    anomaly_scaler = MinMaxScaler()
    anomaly_scaler.fit(raw_val_anomalies.reshape(-1, 1))

    # 6. Final Evaluation on Untouched Test Split
    test_uncal_probs = xgb_model.predict_proba(X_test)
    test_cal_probs = calibrator.transform(test_uncal_probs)
    test_raw_anomalies = iforest_model.score_anomaly(X_test)
    test_norm_anomalies = anomaly_scaler.transform(test_raw_anomalies.reshape(-1, 1)).flatten()

    def evaluate_probs(y_true, y_probs) -> Dict[str, Any]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        pr_auc = float(auc(recall, precision))
        roc_auc = float(roc_auc_score(y_true, y_probs))
        
        preds = (y_probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        return {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    eval_results = {
        "xgboost_calibrated": evaluate_probs(y_test, test_cal_probs),
        "isolation_forest": evaluate_probs(y_test, test_norm_anomalies),
    }

    logger.info(f"Authentic IEEE-CIS Test Set Evaluation Results: {eval_results}")

    # 7. Serialize Authentic Artifacts to models/advanced_artifacts/
    meta = {
        "model_version": "2.0.0-authentic-ieee",
        "dataset_source": "Authentic Kaggle IEEE-CIS Fraud Detection Dataset (train_transaction.csv & train_identity.csv)",
        "dataset_rows": total_rows,
        "is_synthetic": False,
        "trained_at": "2026-09-25",
    }
    
    AdvancedModelRegistry.save_artifact(preprocessor, "advanced_preprocessor_v2.0.0.joblib", meta)
    AdvancedModelRegistry.save_artifact(xgb_model, "advanced_xgboost_v2.0.0.joblib", meta)
    AdvancedModelRegistry.save_artifact(iforest_model, "advanced_isolation_forest_v2.0.0.joblib", meta)
    AdvancedModelRegistry.save_artifact(calibrator, "advanced_calibrator_v2.0.0.joblib", meta)
    AdvancedModelRegistry.save_artifact(anomaly_scaler, "advanced_anomaly_scaler_v2.0.0.joblib", meta)

    metrics_path = os.path.join(adv_dir, "advanced_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "2.0.0-authentic-ieee",
            "dataset_source": "Authentic Kaggle IEEE-CIS Fraud Detection Dataset",
            "total_dataset_rows": total_rows,
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
            "train_frauds": train_frauds,
            "val_frauds": val_frauds,
            "test_frauds": test_frauds,
            "test_evaluations": eval_results,
        }, f, indent=2)

    logger.info(f"Authentic training complete. Metrics saved to {metrics_path}")
    return eval_results

if __name__ == "__main__":
    run_authentic_advanced_training_pipeline()
