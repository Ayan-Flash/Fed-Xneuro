"""
Application configuration settings.
"""

import os
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Fed-XNeuro Simulation Platform"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "fed-xneuro-super-secret-key-change-in-production-2026"
    ALGORITHM: str = "HS256"
    DATABASE_URL: str = "sqlite:///./fed_xneuro.db"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
