"""
Security primitives: password hashing (bcrypt via passlib) and JWT
creation/validation (python-jose). No business logic lives here — this
module only deals with cryptographic concerns.
"""
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.core.config import settings
from src.core.exceptions import InvalidTokenException, TokenExpiredException

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


# --------------------------------------------------------------------------
# JWT creation
# --------------------------------------------------------------------------
def _create_token(subject: str | UUID, token_type: TokenType, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type.value,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: str | UUID) -> str:
    """Create a short-lived JWT access token carrying the user's id as `sub`."""
    return _create_token(
        subject=user_id,
        token_type=TokenType.ACCESS,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str | UUID) -> str:
    """Create a long-lived JWT refresh token."""
    return _create_token(
        subject=user_id,
        token_type=TokenType.REFRESH,
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


# --------------------------------------------------------------------------
# JWT validation
# --------------------------------------------------------------------------
def decode_token(token: str, expected_type: TokenType) -> dict:
    """
    Decode and validate a JWT. Raises InvalidTokenException /
    TokenExpiredException on any failure so callers don't need to know
    about jose's internal exception types.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException()
    except JWTError:
        raise InvalidTokenException()

    if payload.get("type") != expected_type.value:
        raise InvalidTokenException(f"Expected a {expected_type.value} token.")

    if "sub" not in payload:
        raise InvalidTokenException("Token payload missing subject.")

    return payload
