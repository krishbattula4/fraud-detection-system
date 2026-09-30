"""Generic repository interface pattern and concrete SQLAlchemy repositories."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Any, Dict
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.db.models import (
    TransactionRecord,
    PredictionRecord,
    AlertRecord,
    ReviewRecord,
    ModelVersionRecord,
)
from src.schemas.alert import AlertStatus
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger("db_repository")

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract repository for persisting and querying entities."""

    @abstractmethod
    def add(self, entity: T) -> T:
        """Persist new entity."""
        pass

    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Fetch entity by unique primary key ID."""
        pass

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List entities with pagination."""
        pass


class TransactionRepository(BaseRepository[TransactionRecord]):
    """SQLAlchemy repository for TransactionRecord entities."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, entity: TransactionRecord) -> TransactionRecord:
        try:
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to persist TransactionRecord: {str(e)}")
            raise DatabaseError(f"Failed to persist transaction record: {str(e)}")

    def get_by_id(self, entity_id: str) -> Optional[TransactionRecord]:
        try:
            return self.db.query(TransactionRecord).filter(TransactionRecord.id == entity_id).first()
        except Exception as e:
            logger.error(f"Database query error in get_by_id for transaction '{entity_id}': {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def get_by_transaction_id(self, transaction_id: str) -> Optional[TransactionRecord]:
        try:
            return self.db.query(TransactionRecord).filter(TransactionRecord.transaction_id == transaction_id).first()
        except Exception as e:
            logger.error(f"Database query error in get_by_transaction_id '{transaction_id}': {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def list(self, limit: int = 100, offset: int = 0) -> List[TransactionRecord]:
        try:
            return (
                self.db.query(TransactionRecord)
                .order_by(TransactionRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Database query error in list transactions: {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")


class PredictionRepository(BaseRepository[PredictionRecord]):
    """SQLAlchemy repository for PredictionRecord entities."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, entity: PredictionRecord) -> PredictionRecord:
        try:
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to persist PredictionRecord: {str(e)}")
            raise DatabaseError(f"Failed to persist prediction record: {str(e)}")

    def get_by_id(self, entity_id: str) -> Optional[PredictionRecord]:
        try:
            return self.db.query(PredictionRecord).filter(PredictionRecord.id == entity_id).first()
        except Exception as e:
            logger.error(f"Database query error for prediction '{entity_id}': {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def get_by_transaction_record_id(self, transaction_record_id: str) -> Optional[PredictionRecord]:
        try:
            return (
                self.db.query(PredictionRecord)
                .filter(PredictionRecord.transaction_record_id == transaction_record_id)
                .order_by(PredictionRecord.evaluated_at.desc())
                .first()
            )
        except Exception as e:
            logger.error(f"Database query error for prediction by transaction_record_id: {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def list(self, limit: int = 100, offset: int = 0) -> List[PredictionRecord]:
        try:
            return (
                self.db.query(PredictionRecord)
                .order_by(PredictionRecord.evaluated_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Database query error in list predictions: {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")


class AlertRepository(BaseRepository[AlertRecord]):
    """SQLAlchemy repository for AlertRecord entities."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, entity: AlertRecord) -> AlertRecord:
        try:
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to persist AlertRecord: {str(e)}")
            raise DatabaseError(f"Failed to persist alert record: {str(e)}")

    def get_by_id(self, entity_id: str) -> Optional[AlertRecord]:
        try:
            return self.db.query(AlertRecord).filter(AlertRecord.id == entity_id).first()
        except Exception as e:
            logger.error(f"Database query error for alert '{entity_id}': {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def list(
        self,
        status: Optional[str] = None,
        min_risk_score: Optional[float] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AlertRecord]:
        try:
            query = self.db.query(AlertRecord)

            if status:
                query = query.filter(AlertRecord.status == status)
            if min_risk_score is not None:
                query = query.filter(AlertRecord.risk_score >= min_risk_score)
            if risk_level:
                query = query.filter(AlertRecord.risk_level == risk_level)

            return query.order_by(AlertRecord.created_at.desc()).offset(offset).limit(limit).all()
        except Exception as e:
            logger.error(f"Database query error in list alerts: {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def update_review(
        self, alert: AlertRecord, new_status: str, reviewer_notes: Optional[str] = None
    ) -> AlertRecord:
        try:
            alert.status = new_status
            alert.reviewed_at = datetime.now(timezone.utc)
            if reviewer_notes:
                alert.reviewer_notes = reviewer_notes
            self.db.commit()
            self.db.refresh(alert)
            return alert
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update alert review status: {str(e)}")
            raise DatabaseError(f"Failed to update alert review status: {str(e)}")


class ReviewRepository(BaseRepository[ReviewRecord]):
    """SQLAlchemy repository for ReviewRecord entities."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, entity: ReviewRecord) -> ReviewRecord:
        try:
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to persist ReviewRecord: {str(e)}")
            raise DatabaseError(f"Failed to persist review record: {str(e)}")

    def get_by_id(self, entity_id: str) -> Optional[ReviewRecord]:
        try:
            return self.db.query(ReviewRecord).filter(ReviewRecord.id == entity_id).first()
        except Exception as e:
            logger.error(f"Database query error for review '{entity_id}': {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")

    def list(self, limit: int = 100, offset: int = 0) -> List[ReviewRecord]:
        try:
            return self.db.query(ReviewRecord).order_by(ReviewRecord.reviewed_at.desc()).offset(offset).limit(limit).all()
        except Exception as e:
            logger.error(f"Database query error in list reviews: {str(e)}")
            raise DatabaseError(f"Database query error: {str(e)}")


class MetricsRepository:
    """Repository for querying system operational and performance metrics from DB."""

    def __init__(self, db: Session):
        self.db = db

    def get_operational_metrics(self) -> Dict[str, Any]:
        try:
            total_evaluations = self.db.query(func.count(PredictionRecord.id)).scalar() or 0
            total_alerts_triggered = self.db.query(func.count(AlertRecord.id)).scalar() or 0

            # Count per risk tier
            tier_counts_query = (
                self.db.query(PredictionRecord.risk_level, func.count(PredictionRecord.id))
                .group_by(PredictionRecord.risk_level)
                .all()
            )
            evaluations_by_tier = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            for tier, count in tier_counts_query:
                if tier in evaluations_by_tier:
                    evaluations_by_tier[tier] = count

            return {
                "total_evaluations": total_evaluations,
                "total_alerts_triggered": total_alerts_triggered,
                "evaluations_by_tier": evaluations_by_tier,
            }
        except Exception as e:
            logger.error(f"Failed to calculate system operational metrics: {str(e)}")
            raise DatabaseError(f"Failed to calculate system metrics: {str(e)}")
