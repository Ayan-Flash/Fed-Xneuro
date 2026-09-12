"""
Database initialization and metadata helper.
"""

from backend.app.db.session import engine, Base, SessionLocal, get_db


def init_db() -> None:
    """Creates database tables."""
    # Import all models to register with Base.metadata
    from backend.app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
