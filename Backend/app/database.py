"""
app/database.py
---------------
SQLAlchemy engine, session factory, and declarative base.
Supports both PostgreSQL (Neon, production) and SQLite (local / tests).
"""

from __future__ import annotations

import logging
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _get_engine():
    settings = get_settings()
    url = settings.database_url

    if url.startswith("sqlite"):
        # SQLite — used for local dev and tests only
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    else:
        # PostgreSQL (Neon) — production
        engine = create_engine(
            url,
            pool_pre_ping=True,       # detects stale connections
            pool_size=1,              # keep pool small for serverless
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
    """Create all tables that do not yet exist."""
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
