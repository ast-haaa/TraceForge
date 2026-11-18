# agentreplay/db.py
"""Database engine and session factory.

DATABASE_URL defaults to a SQLite file in the project root.
Set the DATABASE_URL environment variable to switch to Postgres.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///e:/Projects/TraceForge/agentreplay.db",
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine_kwargs = {"connect_args": connect_args, "echo": False, "future": True}
if DATABASE_URL in ("sqlite://", "sqlite:///:memory:"):
    engine_kwargs["poolclass"] = StaticPool

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)
