"""
Authentication and user management service.
"""

from typing import Optional
from sqlalchemy.orm import Session
from backend.app.db.models.user import User
from backend.app.schemas.user import UserCreate
from backend.app.core.security import get_password_hash, verify_password, create_access_token
from backend.app.core.exceptions import ValidationException, AuthenticationException


class AuthService:
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def register(db: Session, user_in: UserCreate) -> User:
        existing = AuthService.get_user_by_email(db, user_in.email)
        if existing:
            raise ValidationException("Email is already registered")

        user = User(
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            full_name=user_in.full_name,
            is_active=True,
            is_superuser=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate(db: Session, email: str, password: str) -> str:
        user = AuthService.get_user_by_email(db, email)
        if not user or not verify_password(password, user.hashed_password):
            raise AuthenticationException("Incorrect email or password")
        if not user.is_active:
            raise AuthenticationException("User account is disabled")

        return create_access_token(subject=user.id)
