"""
Database initialization and metadata helper.
"""

from backend.app.db.session import engine, Base, SessionLocal, get_db


def init_db() -> None:
    """Creates database tables and seeds demo test accounts."""
    # Import all models to register with Base.metadata
    from backend.app.db import models  # noqa: F401
    from backend.app.db.models.user import User
    from backend.app.core.security import get_password_hash

    Base.metadata.create_all(bind=engine)

    # Seed default demo users for testing
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "doctor@memorial.org").first():
            clinician = User(
                email="doctor@memorial.org",
                hashed_password=get_password_hash("demo123"),
                full_name="Dr. Sarah Chen, MD",
                is_active=True,
                is_superuser=False,
            )
            db.add(clinician)

        if not db.query(User).filter(User.email == "admin@platform.com").first():
            admin = User(
                email="admin@platform.com",
                hashed_password=get_password_hash("admin123"),
                full_name="Alex Rivera",
                is_active=True,
                is_superuser=True,
            )
            db.add(admin)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error seeding demo users: {e}")
    finally:
        db.close()

