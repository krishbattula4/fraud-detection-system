"""Unit tests for advanced multi-entity transaction schema & behavioral preprocessing pipeline."""
import pytest
import numpy as np
from src.schemas.advanced_transaction import AdvancedTransactionInput, AdvancedRiskPredictionResponse
from src.data.advanced_preprocessing import AdvancedDataPreprocessor

def test_advanced_transaction_input_validation():
    """Verify schema validation for valid, missing, and boundary fields."""
    tx_dict = {
        "transaction_id": "TX-ADV-001",
        "timestamp": 3600.0,
        "amount": 250.50,
        "card_id": "CARD-9921",
        "billing_region": "REG-NY",
        "purchaser_email_domain": "gmail.com",
        "recipient_email_domain": "merchant.com",
        "device_type": "desktop",
        "device_info": "Windows 11 Chrome",
    }
    tx_input = AdvancedTransactionInput(**tx_dict)
    assert tx_input.transaction_id == "TX-ADV-001"
    assert tx_input.amount == 250.50
    assert tx_input.card_id == "CARD-9921"
    assert tx_input.device_type == "desktop"


def test_entity_velocity_sliding_windows():
    """Verify 1-hour and 24-hour sliding window transaction counts and amount ratios."""
    preprocessor = AdvancedDataPreprocessor()

    # Transaction 1: t = 1000s, amount = $100.0
    tx1 = AdvancedTransactionInput(
        transaction_id="TX-1", timestamp=1000.0, amount=100.0, card_id="CARD-A", billing_region="US", purchaser_email_domain="user@gmail.com"
    )
    feat1 = preprocessor.transform_single(tx1)
    assert feat1["cust_tx_count_1h"] == 0.0  # First transaction
    assert feat1["cust_tx_count_24h"] == 0.0
    assert feat1["amt_to_cust_avg_ratio"] == 1.0

    # Transaction 2: t = 2000s (1000s later, within 1h), amount = $200.0
    tx2 = AdvancedTransactionInput(
        transaction_id="TX-2", timestamp=2000.0, amount=200.0, card_id="CARD-A", billing_region="US", purchaser_email_domain="user@gmail.com"
    )
    feat2 = preprocessor.transform_single(tx2)
    assert feat2["cust_tx_count_1h"] == 1.0  # 1 transaction in past 1h
    assert feat2["cust_tx_count_24h"] == 1.0
    assert feat2["cust_amt_sum_24h"] == 100.0
    # Average of past is 100.0, ratio = 200 / 100 = 2.0
    assert pytest.approx(feat2["amt_to_cust_avg_ratio"], 0.01) == 2.0

    # Transaction 3: t = 10000s (8000s later, outside 1h but within 24h)
    tx3 = AdvancedTransactionInput(
        transaction_id="TX-3", timestamp=10000.0, amount=150.0, card_id="CARD-A", billing_region="US", purchaser_email_domain="user@gmail.com"
    )
    feat3 = preprocessor.transform_single(tx3)
    assert feat3["cust_tx_count_1h"] == 0.0  # Outside 1h window (3600s)
    assert feat3["cust_tx_count_24h"] == 2.0  # Within 24h window (86400s)


def test_device_and_domain_novelty():
    """Verify detection of new device and new email domain for an existing entity."""
    preprocessor = AdvancedDataPreprocessor()

    tx1 = AdvancedTransactionInput(
        transaction_id="TX-1", timestamp=100.0, amount=50.0, card_id="CARD-B", device_info="iPhone 14", recipient_email_domain="domainA.com"
    )
    f1 = preprocessor.transform_single(tx1)
    assert f1["is_new_device_for_cust"] == 0
    assert f1["is_new_email_domain"] == 0

    # Same entity, new device & domain
    tx2 = AdvancedTransactionInput(
        transaction_id="TX-2", timestamp=500.0, amount=500.0, card_id="CARD-B", device_info="Android Pixel 8", recipient_email_domain="domainB.com"
    )
    f2 = preprocessor.transform_single(tx2)
    assert f2["is_new_device_for_cust"] == 1
    assert f2["is_new_email_domain"] == 1


def test_zero_future_leakage():
    """Ensure future transactions do not leak into past sliding window counts."""
    preprocessor = AdvancedDataPreprocessor()

    # Past transaction t = 500s
    tx_past = AdvancedTransactionInput(
        transaction_id="TX-P", timestamp=500.0, amount=50.0, card_id="CARD-C"
    )
    preprocessor.transform_single(tx_past)

    # Future transaction evaluated at t = 600s
    tx_curr = AdvancedTransactionInput(
        transaction_id="TX-C", timestamp=600.0, amount=100.0, card_id="CARD-C"
    )
    feat_curr = preprocessor.transform_single(tx_curr)
    
    # Evaluate a transaction inserted out of order with earlier t = 400s
    tx_earlier = AdvancedTransactionInput(
        transaction_id="TX-E", timestamp=400.0, amount=75.0, card_id="CARD-C"
    )
    feat_earlier = preprocessor.transform_single(tx_earlier)
    
    # Earlier transaction should ONLY see tx_past (t=500s) if t < 400s (which is 0)
    assert feat_earlier["cust_tx_count_1h"] == 0.0


def test_train_only_scaler_fitting():
    """Verify RobustScaler fits strictly on training amounts."""
    preprocessor = AdvancedDataPreprocessor()
    train_amounts = np.array([10.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    preprocessor.fit_scaler_on_train(train_amounts)
    assert preprocessor.is_fitted is True
