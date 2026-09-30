"""Unit test suite for TransactionSimulator."""
import pytest
from src.simulation.simulator import TransactionSimulator
from src.schemas.transaction import TransactionInput
from src.data.preprocessor import FEATURE_COLUMNS


def test_generate_single_test_transaction():
    """Test generating a single deterministic test transaction matching dataset schema."""
    sim = TransactionSimulator(seed=123)
    tx = sim.generate_test_transaction(transaction_id="SIM-TEST-01", is_anomaly=False)

    assert isinstance(tx, TransactionInput)
    assert tx.transaction_id == "SIM-TEST-01"
    assert tx.time >= 0.0
    assert tx.amount >= 0.0

    vector = tx.to_feature_vector()
    assert len(vector) == 30
    assert len(FEATURE_COLUMNS) == 30


def test_generate_anomaly_test_transaction():
    """Test generating an anomaly test transaction."""
    sim = TransactionSimulator(seed=456)
    tx = sim.generate_test_transaction(is_anomaly=True)

    assert isinstance(tx, TransactionInput)
    assert tx.amount >= 1000.0  # Anomaly simulation higher amount range


def test_stream_transactions_generator():
    """Test stream_transactions yields valid TransactionInput models."""
    sim = TransactionSimulator(seed=789)
    stream = sim.stream_transactions(rate_per_sec=100.0, count=5)
    txs = list(stream)

    assert len(txs) == 5
    for tx in txs:
        assert isinstance(tx, TransactionInput)
        assert len(tx.to_feature_vector()) == 30
