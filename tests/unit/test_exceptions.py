"""Unit tests for custom exception hierarchy."""
from src.core.exceptions import (
    FraudSystemError,
    ValidationError,
    ModelNotFoundError,
    InferenceError,
    RiskEngineError,
    DatabaseError,
)


def test_fraud_system_error_inheritance():
    """Verify custom exceptions inherit from base FraudSystemError."""
    err = ValidationError("Invalid feature schema", details={"feature": "V1"})
    assert isinstance(err, FraudSystemError)
    assert err.message == "Invalid feature schema"
    assert err.details == {"feature": "V1"}


def test_error_to_dict_format():
    """Verify to_dict serialization format."""
    err = InferenceError("Model evaluation failed", details={"model": "xgboost"})
    d = err.to_dict()
    assert d["error_type"] == "InferenceError"
    assert d["message"] == "Model evaluation failed"
    assert d["details"]["model"] == "xgboost"
