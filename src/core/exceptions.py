"""Custom exception hierarchy for the Fraud Detection System."""
from typing import Optional, Any, Dict


class FraudSystemError(Exception):
    """Base exception class for all domain errors within the system."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception details to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(FraudSystemError):
    """Raised when data input or schema validation fails."""
    pass


class ModelNotFoundError(FraudSystemError):
    """Raised when a requested model artifact is missing or unreadable."""
    pass


class InferenceError(FraudSystemError):
    """Raised when model evaluation or score prediction fails at runtime."""
    pass


class RiskEngineError(FraudSystemError):
    """Raised when risk score aggregation or calibration logic encounters an error."""
    pass


class DatabaseError(FraudSystemError):
    """Raised when database query execution or persistence fails."""
    pass
