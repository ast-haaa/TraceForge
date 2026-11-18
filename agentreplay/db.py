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
    """Create all tables (idempotent) and auto-migrate SQLite columns if needed."""
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite") and not DATABASE_URL.endswith(":memory:"):
        with engine.connect() as conn:
            try:
                res = conn.exec_driver_sql("PRAGMA table_info(generated_tests);")
                cols = {row[1] for row in res}
                if cols:
                    if "file_path" not in cols:
                        conn.exec_driver_sql("ALTER TABLE generated_tests ADD COLUMN file_path TEXT;")
                    if "pytest_output" not in cols:
                        conn.exec_driver_sql("ALTER TABLE generated_tests ADD COLUMN pytest_output TEXT;")
                    if "status" not in cols:
                        conn.exec_driver_sql("ALTER TABLE generated_tests ADD COLUMN status TEXT DEFAULT 'pending';")
                    if "rejection_reason" not in cols:
                        conn.exec_driver_sql("ALTER TABLE generated_tests ADD COLUMN rejection_reason TEXT;")
                    conn.commit()
            except Exception:
                pass
