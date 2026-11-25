"""
app/database.py
───────────────
SQLAlchemy engine, session factory, and declarative base.
All ORM models import Base from here.
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
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        # Required for SQLite to work correctly with FastAPI's thread model.
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        echo=False,           # Set True to log all SQL statements during debug
    )

    # Enable WAL mode for better concurrent read performance on SQLite.
    if settings.database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = _get_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def init_db() -> None:
    """Create all tables that are not yet present in the database."""
    # Import models so SQLAlchemy registers them with Base before create_all.
    import app.models.db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created.")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and closes it afterwards.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
