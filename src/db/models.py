"""SQLAlchemy ORM database models."""
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from src.db.session import Base


class TransactionRecord(Base):
    """Stored transaction entity."""
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, index=True)
    transaction_id = Column(String(100), index=True, nullable=True)
    time = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    raw_features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship("PredictionRecord", back_populates="transaction")


class PredictionRecord(Base):
    """Stored prediction and risk scoring entity."""
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, index=True)
    transaction_record_id = Column(String(36), ForeignKey("transactions.id"), nullable=False)
    risk_score = Column(Float, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)
    decision = Column(String(30), nullable=False)
    xgboost_prob = Column(Float, nullable=False)
    iforest_anomaly = Column(Float, nullable=False)
    lof_anomaly = Column(Float, nullable=False)
    explanations = Column(JSON, nullable=True)
    model_version = Column(String(50), nullable=False)
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    transaction = relationship("TransactionRecord", back_populates="predictions")
    alerts = relationship("AlertRecord", back_populates="prediction")


class AlertRecord(Base):
    """Risk alert entity generated when risk scores exceed threshold."""
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, index=True)
    prediction_id = Column(String(36), ForeignKey("predictions.id"), nullable=False)
    risk_score = Column(Float, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)
    status = Column(String(30), default="OPEN", nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    prediction = relationship("PredictionRecord", back_populates="alerts")
    reviews = relationship("ReviewRecord", back_populates="alert")


class ReviewRecord(Base):
    """Audit log of analyst investigation reviews."""
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True, index=True)
    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=False)
    analyst_id = Column(String(100), nullable=False)
    previous_status = Column(String(30), nullable=False)
    new_status = Column(String(30), nullable=False)
    notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    alert = relationship("AlertRecord", back_populates="reviews")


class ModelVersionRecord(Base):
    """Registry of trained model artifact metadata."""
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    algorithm_name = Column(String(100), nullable=False)
    metrics_summary = Column(JSON, nullable=False)
    artifact_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Integer, default=0, nullable=False)
