"""EDA and statistical profiling utilities for credit card fraud datasets."""
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.core.logging import get_logger

logger = get_logger("data_eda")


class DatasetEDA:
    """Exploratory Data Analysis and statistical summary generator."""

    @staticmethod
    def generate_statistical_profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical properties of features and target class."""
        if df is None or df.empty:
            return {"error": "Empty dataset"}

        total_records = len(df)
        fraud_records = int((df["Class"] == 1).sum()) if "Class" in df.columns else 0
        legit_records = total_records - fraud_records
        fraud_ratio = (fraud_records / total_records) if total_records > 0 else 0.0

        amount_stats = {}
        if "Amount" in df.columns:
            amount_stats = {
                "mean": float(df["Amount"].mean()),
                "std": float(df["Amount"].std()),
                "median": float(df["Amount"].median()),
                "iqr": float(df["Amount"].quantile(0.75) - df["Amount"].quantile(0.25)),
                "min": float(df["Amount"].min()),
                "max": float(df["Amount"].max()),
                "fraud_mean_amount": float(df[df["Class"] == 1]["Amount"].mean()) if fraud_records > 0 else 0.0,
                "legit_mean_amount": float(df[df["Class"] == 0]["Amount"].mean()) if legit_records > 0 else 0.0,
            }

        time_stats = {}
        if "Time" in df.columns:
            time_stats = {
                "min_seconds": float(df["Time"].min()),
                "max_seconds": float(df["Time"].max()),
                "duration_hours": float((df["Time"].max() - df["Time"].min()) / 3600.0),
                "is_monotonic": bool(df["Time"].is_monotonic_increasing),
            }

        pca_summary = {}
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
        if pca_cols:
            means = df[pca_cols].mean().to_dict()
            stds = df[pca_cols].std().to_dict()
            pca_summary = {
                "mean_range": [float(min(means.values())), float(max(means.values()))],
                "std_range": [float(min(stds.values())), float(max(stds.values()))],
            }

        return {
            "total_records": total_records,
            "fraud_records": fraud_records,
            "legit_records": legit_records,
            "fraud_ratio": fraud_ratio,
            "fraud_prevalence_pct": round(fraud_ratio * 100, 5),
            "amount_stats": amount_stats,
            "time_stats": time_stats,
            "pca_summary": pca_summary,
        }
