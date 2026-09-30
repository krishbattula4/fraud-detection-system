"""SQLAlchemy database engine and session factory."""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from src.core.config import get_settings

Base = declarative_base()


def get_db_engine(database_url: str = None):
    """Return SQLAlchemy engine instance."""
    url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def init_db(engine=None) -> None:
    """Initialize database tables for all defined ORM models."""
    db_engine = engine or get_db_engine()
    # Import ORM models to ensure they are registered with Base.metadata
    import src.db.models  # noqa: F401
    Base.metadata.create_all(bind=db_engine)


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session instances."""
    engine = get_db_engine()
    init_db(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
