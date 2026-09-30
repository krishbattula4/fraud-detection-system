"""Pytest configuration and shared fixtures."""
import pytest
from src.core.config import Settings
from src.schemas.transaction import TransactionInput


@pytest.fixture
def sample_settings():
    """Return isolated test settings instance."""
    return Settings(
        environment="testing",
        log_level="DEBUG",
        database_url="sqlite:///:memory:",
        api_host="127.0.0.1",
        api_port=8000,
    )


@pytest.fixture
def sample_transaction_input():
    """Return valid sample TransactionInput payload matching ULB feature format."""
    return TransactionInput(
        transaction_id="tx_test_001",
        time=100.0,
        amount=250.50,
        v1=-1.35, v2=1.20, v3=-0.50, v4=0.80, v5=-0.30, v6=0.10, v7=0.45, v8=0.25,
        v9=-0.15, v10=0.60, v11=-0.80, v12=0.90, v13=-0.40, v14=-1.10, v15=0.05, v16=0.35,
        v17=-0.70, v18=0.40, v19=0.15, v20=-0.10, v21=0.05, v22=-0.20, v23=0.10, v24=-0.30,
        v25=0.25, v26=-0.15, v27=0.08, v28=-0.02
    )
