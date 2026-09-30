"""Data validator abstract contract and concrete implementation for schema verification and dataset quality analysis."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger("data_validator")

EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_COLUMNS = EXPECTED_FEATURES + ["Class"]


class BaseDataValidator(ABC):
    """Abstract interface for validating input datasets and transaction payloads."""

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate column names, types, and required features."""
        pass

    @abstractmethod
    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values, infinite values, and unexpected negative amounts."""
        pass


class DataValidator(BaseDataValidator):
    """Concrete data validator for ULB Credit Card Fraud dataset schema."""

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate column presence, expected features, and data types."""
        errors: List[str] = []

        if df is None or df.empty:
            errors.append("Dataset DataFrame is empty or None.")
            return False, errors

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required dataset columns: {missing_cols}")

        non_numeric_cols = [
            col for col in df.columns if col in EXPECTED_COLUMNS and not np.issubdtype(df[col].dtype, np.number)
        ]
        if non_numeric_cols:
            errors.append(f"Non-numeric data type detected in columns: {non_numeric_cols}")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Schema validation failed with errors: {errors}")

        return is_valid, errors

    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality inspection."""
        is_valid, errors = self.validate_schema(df)
        if not is_valid:
            raise ValidationError(f"Cannot perform quality check on invalid schema: {errors}", details={"errors": errors})

        total_rows = len(df)
        total_cols = len(df.columns)
        null_counts = df[EXPECTED_COLUMNS].isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())

        # Check infinite values
        inf_counts = {col: int(np.isinf(df[col]).sum()) for col in EXPECTED_COLUMNS}
        total_infs = sum(inf_counts.values())

        # Check negative amounts
        negative_amounts = int((df["Amount"] < 0).sum()) if "Amount" in df.columns else 0

        # Duplicate row count
        duplicate_rows = int(df.duplicated().sum())

        # Class target inspection
        class_counts = df["Class"].value_counts().to_dict()
        unique_classes = df["Class"].unique()
        invalid_classes = [c for c in unique_classes if c not in [0, 1]]

        if invalid_classes:
            raise ValidationError(
                f"Target 'Class' column contains invalid label values: {invalid_classes}. Expected binary labels [0, 1].",
                details={"invalid_classes": invalid_classes}
            )

        fraud_count = int(class_counts.get(1, 0))
        non_fraud_count = int(class_counts.get(0, 0))
        fraud_percentage = (fraud_count / total_rows * 100) if total_rows > 0 else 0.0

        min_time = float(df["Time"].min()) if "Time" in df.columns else 0.0
        max_time = float(df["Time"].max()) if "Time" in df.columns else 0.0
        min_amount = float(df["Amount"].min()) if "Amount" in df.columns else 0.0
        max_amount = float(df["Amount"].max()) if "Amount" in df.columns else 0.0

        warnings: List[str] = []
        if total_nulls > 0:
            warnings.append(f"Found {total_nulls} missing/null values across features.")
        if total_infs > 0:
            warnings.append(f"Found {total_infs} infinite numerical values.")
        if negative_amounts > 0:
            warnings.append(f"Found {negative_amounts} transactions with negative dollar amounts.")
        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate rows in dataset.")
        if fraud_count == 0:
            warnings.append("No positive fraud transactions (Class=1) found in dataset.")

        quality_report = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_nulls": total_nulls,
            "null_counts": null_counts,
            "total_infinite": total_infs,
            "infinite_counts": inf_counts,
            "negative_amounts": negative_amounts,
            "duplicate_rows": duplicate_rows,
            "fraud_count": fraud_count,
            "non_fraud_count": non_fraud_count,
            "fraud_percentage": round(fraud_percentage, 5),
            "min_time": min_time,
            "max_time": max_time,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "warnings": warnings,
        }

        logger.info(
            f"Quality check complete: {total_rows} rows, {fraud_count} fraud ({fraud_percentage:.3f}%), "
            f"{total_nulls} nulls, {duplicate_rows} duplicates."
        )

        return quality_report
