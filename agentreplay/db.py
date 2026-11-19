# agentreplay/db.py
"""Database engine and session factory.

DATABASE_URL defaults to a SQLite file in the project root.
Set the DATABASE_URL environment variable to switch to Postgres.
"""
import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)

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

                res_eval = conn.exec_driver_sql("PRAGMA table_info(eval_labels);")
                eval_cols = {row[1] for row in res_eval}
                if eval_cols:
                    if "ground_truth_node" not in eval_cols:
                        conn.exec_driver_sql("ALTER TABLE eval_labels ADD COLUMN ground_truth_node TEXT;")
                    conn.commit()

                res_inv = conn.exec_driver_sql("PRAGMA table_info(investigations);")
                inv_cols = {row[1] for row in res_inv}
                if inv_cols:
                    if "summary" not in inv_cols:
                        conn.exec_driver_sql("ALTER TABLE investigations ADD COLUMN summary TEXT;")
                    if "model" not in inv_cols:
                        conn.exec_driver_sql("ALTER TABLE investigations ADD COLUMN model TEXT;")
                    if "prompt_tokens" not in inv_cols:
                        conn.exec_driver_sql("ALTER TABLE investigations ADD COLUMN prompt_tokens INTEGER DEFAULT 0;")
                    if "completion_tokens" not in inv_cols:
                        conn.exec_driver_sql("ALTER TABLE investigations ADD COLUMN completion_tokens INTEGER DEFAULT 0;")
                    if "latency_ms" not in inv_cols:
                        conn.exec_driver_sql("ALTER TABLE investigations ADD COLUMN latency_ms FLOAT DEFAULT 0.0;")
                    conn.commit()
            except Exception as exc:
                logger.warning(
                    "SQLite dev auto-migration encountered an error: %s",
                    exc,
                    exc_info=True,
                )
