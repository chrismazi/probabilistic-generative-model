"""Database connection and session management."""

from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings


@lru_cache
def get_engine() -> Engine:
    """
    Get cached SQLAlchemy engine.
    
    Uses connection pooling with configurable pool size.
    """
    engine = create_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        pool_pre_ping=True,  # Verify connections before use
        echo=False,
    )
    return engine


@lru_cache
def get_session_factory() -> sessionmaker:
    """Get cached session factory."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_session() as session:
            session.execute(...)
            session.commit()
    
    Automatically rolls back on exception.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """
    Dispose of connection pool.
    
    Call this during shutdown or testing.
    """
    get_engine.cache_clear()
    get_session_factory.cache_clear()
