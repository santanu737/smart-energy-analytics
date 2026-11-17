"""Login and JWT handling for the demo user.

Credentials come from .env (SECRET_KEY, DEMO_USER, DEMO_PASSWORD,
ACCESS_TOKEN_EXPIRE_MINUTES). Values are read when used, not at import time.
"""
from __future__ import annotations

import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src import config

load_dotenv(config.PROJECT_ROOT / ".env")

ALGORITHM = "HS256"
bearer_scheme = HTTPBearer(auto_error=False)


def _secret() -> str:
    return os.getenv("SECRET_KEY", "change_me")


def security_warnings() -> list[str]:
    """Messages to print at startup when the .env values are unsafe."""
    out = []
    if _secret() in ("", "change_me") or len(_secret()) < 16:
        out.append("SECRET_KEY is missing, 'change_me' or shorter than 16 characters. "
                   "Anyone could forge a token. Set a long random value in .env.")
    if os.getenv("DEMO_PASSWORD", "change_me") in ("", "change_me", "admin123"):
        out.append("DEMO_PASSWORD is still a default value. Change it in .env.")
    return out


def authenticate_user(username: str, password: str) -> dict | None:
    """Constant-time comparison of both fields (no early exit)."""
    user_ok = hmac.compare_digest(username.encode(), os.getenv("DEMO_USER", "admin").encode())
    pass_ok = hmac.compare_digest(password.encode(), os.getenv("DEMO_PASSWORD", "change_me").encode())
    return {"username": username} if (user_ok and pass_ok) else None


def create_access_token(username: str) -> str:
    minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    now = datetime.now(timezone.utc)
    payload = {"sub": username, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> str:
    unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                 detail="Missing or invalid token",
                                 headers={"WWW-Authenticate": "Bearer"})
    if creds is None:
        raise unauthorized
    try:
        payload = jwt.decode(creds.credentials, _secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise unauthorized
    if not payload.get("sub"):
        raise unauthorized
    return payload["sub"]
