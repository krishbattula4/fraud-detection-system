"""Data loader abstract contract and concrete implementation for CSV loading and chronological splitting."""
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd
from src.core.config import get_settings
from src.core.exceptions import ModelNotFoundError, ValidationError
from src.core.logging import get_logger

logger = get_logger("data_loader")


class BaseDataLoader(ABC):
    """Abstract interface for loading and partitioning dataset sources."""

    @abstractmethod
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw DataFrame from storage path."""
        pass

    @abstractmethod
    def split_chronological(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset chronologically based on Time column to prevent data leakage."""
        pass


class DataLoader(BaseDataLoader):
    """Concrete data loader for credit card fraud datasets."""

    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw DataFrame from configured storage path safely."""
        target_path = file_path or get_settings().raw_dataset_path
        
        if not os.path.exists(target_path):
            logger.error(f"Dataset file not found at path: '{target_path}'")
            raise ModelNotFoundError(
                f"Raw dataset file not found at path: '{target_path}'. Please configure RAW_DATASET_PATH.",
                details={"file_path": target_path}
            )
            
        try:
            logger.info(f"Loading raw dataset from '{target_path}'...")
            df = pd.read_csv(target_path)
            if df.empty:
                raise ValidationError("Loaded dataset is empty (0 rows).", details={"file_path": target_path})
            logger.info(f"Successfully loaded dataset with shape {df.shape}.")
            return df
        except Exception as e:
            if isinstance(e, (ModelNotFoundError, ValidationError)):
                raise e
            logger.error(f"Failed to read CSV file at '{target_path}': {str(e)}")
            raise ValidationError(
                f"Failed to parse CSV dataset: {str(e)}",
                details={"file_path": target_path, "error": str(e)}
            )

    def split_chronological(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset chronologically based on Time column to prevent data leakage.
        
        Default splits: 70% Train, 15% Validation, 15% Test.
        Guarantees: max(train Time) <= min(val Time) <= max(val Time) <= min(test Time).
        """
        if "Time" not in df.columns:
            raise ValidationError("Dataset missing required 'Time' column for chronological splitting.")
            
        if len(df) < 10:
            raise ValidationError(f"Dataset row count ({len(df)}) is too small for 3-way chronological split.")

        if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0) or (train_ratio + val_ratio >= 1.0):
            raise ValidationError("Invalid train/validation ratio parameters. Ratios must sum to less than 1.0.")

        # Ensure sorted by Time
        df_sorted = df.sort_values(by="Time").reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        # Verify temporal non-overlap boundaries
        if not train_df.empty and not val_df.empty:
            assert train_df["Time"].max() <= val_df["Time"].min(), "Train max time exceeds validation min time!"
        if not val_df.empty and not test_df.empty:
            assert val_df["Time"].max() <= test_df["Time"].min(), "Validation max time exceeds test min time!"

        logger.info(
            f"Chronological split completed: Train={len(train_df)} rows ({train_ratio*100:.1f}%), "
            f"Val={len(val_df)} rows ({val_ratio*100:.1f}%), Test={len(test_df)} rows ({(1-train_ratio-val_ratio)*100:.1f}%)."
        )

        return train_df, val_df, test_df
