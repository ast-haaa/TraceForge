# tests/conftest.py
"""
Shared pytest configuration.

IMPORTANT: DATABASE_URL must be set to an in-memory SQLite URL BEFORE any
agentreplay module that calls init_db() is imported.  We set it here at
module load time (before any test-module imports run) so the in-process
engine never touches the real file-based DB.
"""
import os

# Override the database URL to in-memory SQLite for all tests.
# This module is loaded first by pytest (before test modules).
os.environ["DATABASE_URL"] = "sqlite://"

# Now safe to import agentreplay
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from agentreplay.models import Base


@pytest.fixture(scope="function")
def db_session():
    """Fresh in-memory SQLite session for schema/ORM tests."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(engine)
