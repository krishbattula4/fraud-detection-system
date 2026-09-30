"""Transaction input validation models matching ULB Credit Card dataset schema."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Input payload for a single transaction inference evaluation."""
    transaction_id: Optional[str] = Field(default=None, description="Optional client transaction identifier")
    time: float = Field(..., description="Seconds elapsed since first transaction in dataset", ge=0.0)
    amount: float = Field(..., description="Transaction dollar amount", ge=0.0)
    v1: float = Field(..., description="PCA feature V1")
    v2: float = Field(..., description="PCA feature V2")
    v3: float = Field(..., description="PCA feature V3")
    v4: float = Field(..., description="PCA feature V4")
    v5: float = Field(..., description="PCA feature V5")
    v6: float = Field(..., description="PCA feature V6")
    v7: float = Field(..., description="PCA feature V7")
    v8: float = Field(..., description="PCA feature V8")
    v9: float = Field(..., description="PCA feature V9")
    v10: float = Field(..., description="PCA feature V10")
    v11: float = Field(..., description="PCA feature V11")
    v12: float = Field(..., description="PCA feature V12")
    v13: float = Field(..., description="PCA feature V13")
    v14: float = Field(..., description="PCA feature V14")
    v15: float = Field(..., description="PCA feature V15")
    v16: float = Field(..., description="PCA feature V16")
    v17: float = Field(..., description="PCA feature V17")
    v18: float = Field(..., description="PCA feature V18")
    v19: float = Field(..., description="PCA feature V19")
    v20: float = Field(..., description="PCA feature V20")
    v21: float = Field(..., description="PCA feature V21")
    v22: float = Field(..., description="PCA feature V22")
    v23: float = Field(..., description="PCA feature V23")
    v24: float = Field(..., description="PCA feature V24")
    v25: float = Field(..., description="PCA feature V25")
    v26: float = Field(..., description="PCA feature V26")
    v27: float = Field(..., description="PCA feature V27")
    v28: float = Field(..., description="PCA feature V28")

    def to_feature_vector(self) -> List[float]:
        """Return features ordered strictly matching ML pipeline expectation [Time, V1..V28, Amount]."""
        return [
            self.time,
            self.v1, self.v2, self.v3, self.v4, self.v5, self.v6, self.v7, self.v8,
            self.v9, self.v10, self.v11, self.v12, self.v13, self.v14, self.v15, self.v16,
            self.v17, self.v18, self.v19, self.v20, self.v21, self.v22, self.v23, self.v24,
            self.v25, self.v26, self.v27, self.v28,
            self.amount,
        ]


class TransactionBatch(BaseModel):
    """Batch of transaction inputs for evaluation."""
    transactions: List[TransactionInput] = Field(..., min_length=1, description="List of transaction payloads")
