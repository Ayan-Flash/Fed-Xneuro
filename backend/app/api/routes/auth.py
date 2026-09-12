"""
Authentication API routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.schemas.auth import LoginRequest, Token
from backend.app.schemas.user import UserCreate, UserResponse
from backend.app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user account."""
    return AuthService.register(db, user_in)


@router.post("/login", response_model=Token)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates user and issues JWT token."""
    token = AuthService.authenticate(db, login_data.email, login_data.password)
    return {"access_token": token, "token_type": "bearer"}
