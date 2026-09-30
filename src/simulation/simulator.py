"""Abstract and concrete transaction simulator implementation generating test credit-card transaction payloads."""
import time
import uuid
from abc import ABC, abstractmethod
import numpy as np
from typing import Generator, Optional, Dict, Any

from src.schemas.transaction import TransactionInput
from src.core.logging import get_logger

logger = get_logger("transaction_simulator")


class BaseTransactionSimulator(ABC):
    """Abstract interface for streaming test transactions to API endpoints."""

    @abstractmethod
    def stream_transactions(self, rate_per_sec: float = 1.0) -> Generator[TransactionInput, None, None]:
        """Yield realistic test transactions for API simulation testing."""
        pass


class TransactionSimulator(BaseTransactionSimulator):
    """Concrete transaction simulator for generating test credit-card transaction payloads.
    
    CRITICAL DISCLOSURE:
    Generated test transactions are synthetic vectors created strictly for demo/pipeline verification
    and do not represent real financial transactions or real ground-truth fraud labels.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_test_transaction(
        self,
        transaction_id: Optional[str] = None,
        is_anomaly: bool = False,
    ) -> TransactionInput:
        """Generate a single deterministic test credit-card transaction payload matching dataset schema [Time, V1..V28, Amount]."""
        tx_id = transaction_id or f"SIM-TX-{uuid.uuid4().hex[:8].upper()}"
        time_val = float(round(self.rng.uniform(0.0, 172800.0), 2))

        if is_anomaly:
            # Outlier/anomaly simulation vector with larger feature magnitudes
            amount_val = float(round(self.rng.uniform(1500.0, 8500.0), 2))
            v_values = self.rng.normal(loc=2.5, scale=3.0, size=28)
        else:
            # Baseline test transaction vector
            amount_val = float(round(self.rng.uniform(5.0, 350.0), 2))
            v_values = self.rng.normal(loc=0.0, scale=1.0, size=28)

        payload_dict = {
            "transaction_id": tx_id,
            "time": time_val,
            "amount": amount_val,
        }
        for i in range(1, 29):
            payload_dict[f"v{i}"] = float(round(v_values[i - 1], 4))

        return TransactionInput(**payload_dict)

    def stream_transactions(
        self, rate_per_sec: float = 1.0, count: int = 10
    ) -> Generator[TransactionInput, None, None]:
        """Preserves BaseTransactionSimulator contract for streaming test transactions."""
        delay = 1.0 / max(rate_per_sec, 0.1)
        for i in range(count):
            is_anomaly = bool(i % 5 == 0)
            yield self.generate_test_transaction(is_anomaly=is_anomaly)
            time.sleep(delay)
