"""
FastAPI dependency injections.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models.user import User
from backend.app.core.config import settings
from backend.app.core.exceptions import AuthenticationException

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login", auto_error=False)


def get_current_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[User]:
    """Resolves authenticated user from JWT token (if provided)."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationException("Invalid authentication credentials")
    except JWTError:
        raise AuthenticationException("Invalid authentication credentials")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise AuthenticationException("User not found")
    return user


def get_current_active_user(current_user: Optional[User] = Depends(get_current_user)) -> User:
    if not current_user:
        raise AuthenticationException("Authentication required")
    if not current_user.is_active:
        raise AuthenticationException("Inactive user")
    return current_user
