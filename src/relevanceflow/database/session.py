from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


def get_database_url() -> str:
    """Return the configured database URL."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/relevanceflow",
    )


def create_database_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine."""
    url = database_url or get_database_url()

    if not url:
        raise ValueError("DATABASE_URL must not be empty.")

    return create_engine(
        url,
        pool_pre_ping=True,
    )


def create_session_factory(database_url: str | None = None):
    """Create a SQLAlchemy session factory."""
    engine = create_database_engine(database_url)

    return engine, sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )


engine, SessionLocal = create_session_factory()


def get_session() -> Generator[Session, None, None]:
    """Yield a database session and close it afterward."""
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
