"""
app/database.py
---------------
SQLAlchemy engine, session factory, and declarative base.
Database: Neon PostgreSQL (production and local).
Tests create their own isolated in-memory databases and never touch this module.
"""

from __future__ import annotations

import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _get_engine():
    settings = get_settings()

    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,    # drops stale connections before use
        pool_size=1,           # small pool — safe for Vercel serverless
        max_overflow=0,
        echo=False,
    )
    return engine


engine = _get_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def init_db() -> None:
    """Create all tables that do not yet exist in the Neon database."""
    import app.models.db_models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
