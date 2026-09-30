"""Advanced Feature Pipeline & Stateful Entity Behavioral Velocity Engine."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler
from src.schemas.advanced_transaction import AdvancedTransactionInput
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger("advanced_preprocessing")

class AdvancedDataPreprocessor:
    """Leakage-safe feature engineering pipeline for multi-entity transactions."""

    FEATURE_LINEAGE = {
        "amount": {"source": "raw", "transformation": "none", "type": "numeric", "leakage_status": "safe"},
        "amount_scaled": {"source": "derived", "transformation": "RobustScaler(Train-Fitted)", "type": "numeric", "leakage_status": "safe"},
        "cust_tx_count_1h": {"source": "engineered", "transformation": "Past 3600s Sliding Window", "type": "numeric", "leakage_status": "safe"},
        "cust_tx_count_24h": {"source": "engineered", "transformation": "Past 86400s Sliding Window", "type": "numeric", "leakage_status": "safe"},
        "cust_amt_sum_24h": {"source": "engineered", "transformation": "Past 86400s Amount Sum", "type": "numeric", "leakage_status": "safe"},
        "amt_to_cust_avg_ratio": {"source": "engineered", "transformation": "Amount / 30d Mean", "type": "numeric", "leakage_status": "safe"},
        "is_new_device_for_cust": {"source": "engineered", "transformation": "Entity Device History Lookup", "type": "binary", "leakage_status": "safe"},
        "is_new_email_domain": {"source": "engineered", "transformation": "Entity Domain History Lookup", "type": "binary", "leakage_status": "safe"},
        "hour_of_day": {"source": "derived", "transformation": "(timestamp // 3600) % 24", "type": "categorical", "leakage_status": "safe"},
        "day_of_week": {"source": "derived", "transformation": "(timestamp // 86400) % 7", "type": "categorical", "leakage_status": "safe"},
    }

    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
        # Entity state store: entity_id -> List[Tuple[timestamp, amount, device_info, email_domain]]
        self.entity_history: Dict[str, List[Tuple[float, float, str, str]]] = {}

    def construct_entity_id(self, tx: AdvancedTransactionInput) -> str:
        """Construct composite customer entity key from card_id, billing_region, and email domain."""
        card = tx.card_id or "CARD-UNKNOWN"
        region = tx.billing_region or "REG-UNKNOWN"
        email = tx.purchaser_email_domain or "DOM-UNKNOWN"
        return f"{card}_{region}_{email}"

    def compute_behavioral_features(self, tx: AdvancedTransactionInput) -> Dict[str, Any]:
        """Compute sliding-window velocity and novelty features for a transaction strictly using past history (t < current_t)."""
        entity_id = self.construct_entity_id(tx)
        curr_t = float(tx.timestamp)
        curr_amt = float(tx.amount)
        curr_dev = str(tx.device_info or tx.device_type or "DEV-UNKNOWN")
        curr_dom = str(tx.recipient_email_domain or tx.purchaser_email_domain or "DOM-UNKNOWN")

        history = self.entity_history.get(entity_id, [])

        # Filter strictly past transactions: t_prev < curr_t (zero future leakage)
        past_txs = [h for h in history if h[0] < curr_t]

        # 1-hour window (3,600s)
        txs_1h = [h for h in past_txs if curr_t - h[0] <= 3600.0]
        count_1h = len(txs_1h)

        # 24-hour window (86,400s)
        txs_24h = [h for h in past_txs if curr_t - h[0] <= 86400.0]
        count_24h = len(txs_24h)
        sum_amt_24h = sum(h[1] for h in txs_24h)

        # 30-day baseline amount mean (2,592,000s)
        txs_30d = [h for h in past_txs if curr_t - h[0] <= 2592000.0]
        if txs_30d:
            avg_amt_30d = sum(h[1] for h in txs_30d) / len(txs_30d)
            amt_ratio = curr_amt / (avg_amt_30d + 1e-5)
        else:
            amt_ratio = 1.0  # Baseline default for first transaction

        # Device & Domain novelty
        known_devices = {h[2] for h in past_txs if h[2] != "DEV-UNKNOWN"}
        is_new_device = 1 if (curr_dev != "DEV-UNKNOWN" and curr_dev not in known_devices and len(known_devices) > 0) else 0

        known_domains = {h[3] for h in past_txs if h[3] != "DOM-UNKNOWN"}
        is_new_domain = 1 if (curr_dom != "DOM-UNKNOWN" and curr_dom not in known_domains and len(known_domains) > 0) else 0

        # Temporal features
        hour_of_day = int((curr_t // 3600.0) % 24)
        day_of_week = int((curr_t // 86400.0) % 7)

        # Update entity history state store
        if entity_id not in self.entity_history:
            self.entity_history[entity_id] = []
        self.entity_history[entity_id].append((curr_t, curr_amt, curr_dev, curr_dom))

        return {
            "entity_id": entity_id,
            "cust_tx_count_1h": float(count_1h),
            "cust_tx_count_24h": float(count_24h),
            "cust_amt_sum_24h": float(sum_amt_24h),
            "amt_to_cust_avg_ratio": round(float(amt_ratio), 4),
            "is_new_device_for_cust": int(is_new_device),
            "is_new_email_domain": int(is_new_domain),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
        }

    def transform_single(self, tx: AdvancedTransactionInput) -> Dict[str, Any]:
        """Process a single transaction payload into an advanced feature dictionary."""
        behavioral = self.compute_behavioral_features(tx)
        
        # Scaling amount
        raw_amt = float(tx.amount)
        if self.is_fitted:
            amt_scaled = float(self.scaler.transform([[raw_amt]])[0][0])
        else:
            amt_scaled = raw_amt / 100.0  # Fallback prior to fitting

        features = {
            "time": float(tx.timestamp),
            "amount": raw_amt,
            "amount_scaled": amt_scaled,
            **behavioral,
        }

        # Include counting & timedelta features if available
        if tx.counting_features:
            for k, v in tx.counting_features.items():
                features[k.lower()] = float(v)
        if tx.timedelta_features:
            for k, v in tx.timedelta_features.items():
                features[k.lower()] = float(v)

        return features

    def fit_scaler_on_train(self, train_amounts: np.ndarray) -> None:
        """Fit RobustScaler strictly on training amounts."""
        if train_amounts.ndim == 1:
            train_amounts = train_amounts.reshape(-1, 1)
        self.scaler.fit(train_amounts)
        self.is_fitted = True
        logger.info(f"AdvancedDataPreprocessor RobustScaler fitted on {len(train_amounts)} train samples.")
