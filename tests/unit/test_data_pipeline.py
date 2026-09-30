"""Unit test suite for Phase 2 data ingestion, validation, splitting, and preprocessor leakage checks."""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.data.eda import DatasetEDA
from src.data.artifacts import DatasetArtifactManager
from src.core.exceptions import ModelNotFoundError, ValidationError


@pytest.fixture
def sample_raw_dataframe():
    """Create a deterministic synthetic dataset DataFrame matching ULB credit card schema."""
    np.random.seed(42)
    n_samples = 100

    time_vals = np.sort(np.random.uniform(0, 10000, n_samples))
    amount_vals = np.random.exponential(scale=100, size=n_samples)
    
    # 95% non-fraud (0), 5% fraud (1)
    class_vals = np.zeros(n_samples, dtype=int)
    class_vals[:5] = 1

    data = {"Time": time_vals}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(loc=0, scale=1, size=n_samples)
    data["Amount"] = amount_vals
    data["Class"] = class_vals

    return pd.DataFrame(data)


# =====================================================================
# DataLoader Unit Tests
# =====================================================================

def test_loader_missing_file_raises_not_found():
    """Verify loading a non-existent CSV file raises ModelNotFoundError."""
    loader = DataLoader()
    with pytest.raises(ModelNotFoundError):
        loader.load_raw_data("C:/non_existent_directory/missing_dataset.csv")


def test_loader_malformed_csv_raises_validation_error(tmp_path):
    """Verify loading an empty or malformed CSV raises ValidationError."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    
    loader = DataLoader()
    with pytest.raises(ValidationError):
        loader.load_raw_data(str(empty_csv))


def test_loader_valid_csv_loading(tmp_path, sample_raw_dataframe):
    """Verify valid CSV loading succeeds."""
    csv_file = tmp_path / "test_creditcard.csv"
    sample_raw_dataframe.to_csv(csv_file, index=False)

    loader = DataLoader()
    df = loader.load_raw_data(str(csv_file))
    assert len(df) == 100
    assert "Time" in df.columns
    assert "Class" in df.columns


def test_loader_chronological_split_boundaries(sample_raw_dataframe):
    """Verify chronological split respects ratios and temporal non-overlap."""
    loader = DataLoader()
    train_df, val_df, test_df = loader.split_chronological(sample_raw_dataframe, train_ratio=0.7, val_ratio=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert len(train_df) + len(val_df) + len(test_df) == 100

    # Temporal non-overlap check
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()

    # Zero row index overlap check
    assert len(set(train_df.index).intersection(set(val_df.index))) == 0
    assert len(set(val_df.index).intersection(set(test_df.index))) == 0


# =====================================================================
# DataValidator Unit Tests
# =====================================================================

def test_validator_schema_missing_columns(sample_raw_dataframe):
    """Verify validator flags missing required columns."""
    df_incomplete = sample_raw_dataframe.drop(columns=["V14", "Amount"])
    validator = DataValidator()
    is_valid, errors = validator.validate_schema(df_incomplete)
    assert is_valid is False
    assert any("V14" in err or "Amount" in err for err in errors)


def test_validator_quality_reports_metrics(sample_raw_dataframe):
    """Verify quality check reports row counts, target prevalence, and duplicates."""
    validator = DataValidator()
    report = validator.validate_quality(sample_raw_dataframe)
    assert report["total_rows"] == 100
    assert report["fraud_count"] == 5
    assert report["non_fraud_count"] == 95
    assert report["fraud_percentage"] == 5.0
    assert report["total_nulls"] == 0


def test_validator_invalid_target_label_raises_error(sample_raw_dataframe):
    """Verify non-binary target labels raise ValidationError."""
    df_bad = sample_raw_dataframe.copy()
    df_bad.loc[0, "Class"] = 99  # Invalid label
    validator = DataValidator()
    with pytest.raises(ValidationError):
        validator.validate_quality(df_bad)


# =====================================================================
# DataPreprocessor & Leakage Prevention Tests
# =====================================================================

def test_preprocessor_unfitted_transform_raises_error(sample_raw_dataframe):
    """Verify calling transform() before fit() raises ValidationError."""
    preprocessor = DataPreprocessor()
    with pytest.raises(ValidationError):
        preprocessor.transform(sample_raw_dataframe)


def test_preprocessor_leakage_safety(sample_raw_dataframe):
    """Verify preprocessor fits strictly on training split without data leakage."""
    loader = DataLoader()
    train_df, val_df, _ = loader.split_chronological(sample_raw_dataframe)

    preprocessor = DataPreprocessor()
    
    # Fit on train split
    preprocessor.fit(train_df)
    assert preprocessor.is_fitted is True

    # Record scaler center parameters after fitting on train
    train_time_center = preprocessor.time_scaler.center_[0]
    train_amount_center = preprocessor.amount_scaler.center_[0]

    # Transform validation split using pre-fitted state
    X_val_scaled = preprocessor.transform(val_df)
    assert isinstance(X_val_scaled, np.ndarray)
    assert X_val_scaled.shape == (len(val_df), 30)

    # Verify scaler parameters did NOT change after transforming val_df (no leakage!)
    assert preprocessor.time_scaler.center_[0] == train_time_center
    assert preprocessor.amount_scaler.center_[0] == train_amount_center


def test_preprocessor_serialization_round_trip(tmp_path, sample_raw_dataframe):
    """Verify preprocessor state serialization and deserialization via joblib."""
    preprocessor = DataPreprocessor()
    preprocessor.fit(sample_raw_dataframe)
    
    artifact_file = tmp_path / "preprocessor.joblib"
    preprocessor.save_state(str(artifact_file))
    
    # Load into new instance
    loaded_preprocessor = DataPreprocessor()
    loaded_preprocessor.load_state(str(artifact_file))

    assert loaded_preprocessor.is_fitted is True
    
    # Compare transform outputs
    X_orig = preprocessor.transform(sample_raw_dataframe)
    X_loaded = loaded_preprocessor.transform(sample_raw_dataframe)
    np.testing.assert_allclose(X_orig, X_loaded)


# =====================================================================
# EDA & Artifact Manager Unit Tests
# =====================================================================

def test_dataset_eda_profile(sample_raw_dataframe):
    """Verify EDA profiling produces statistical dictionary."""
    profile = DatasetEDA.generate_statistical_profile(sample_raw_dataframe)
    assert profile["total_records"] == 100
    assert profile["fraud_records"] == 5
    assert "amount_stats" in profile
    assert profile["amount_stats"]["min"] >= 0.0


def test_artifact_manager_save_and_load(tmp_path):
    """Verify dataset artifact manager JSON saving and loading."""
    manager = DatasetArtifactManager(output_dir=str(tmp_path))
    data = {"split_name": "train", "rows": 70, "fraud_count": 5}
    filepath = manager.save_split_metadata(data, "test_metadata.json")
    
    loaded = manager.load_metadata("test_metadata.json")
    assert loaded["rows"] == 70
    assert loaded["fraud_count"] == 5
