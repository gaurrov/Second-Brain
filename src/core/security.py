"""
Security primitives: password hashing (bcrypt) and JWT
creation/validation (python-jose). No business logic lives here — this
module only deals with cryptographic concerns.
"""
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID

import bcrypt
from jose import JWTError, jwt

from src.core.config import settings
from src.core.exceptions import InvalidTokenException, TokenExpiredException

# bcrypt operates on bytes and has a hard 72-BYTE input limit (not 72
# characters — a multi-byte UTF-8 password can exceed this well under
# 72 characters). bcrypt>=4.1 raises ValueError past this limit instead
# of silently truncating like older versions / passlib's CryptContext
# did. We truncate explicitly here to preserve that old, safe behavior
# consistently between hashing and verification, rather than letting a
# long password 500 the request.
_BCRYPT_MAX_BYTES = 72


def _prepare_password_bytes(plain_password: str) -> bytes:
    return plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage."""
    return bcrypt.hashpw(
        _prepare_password_bytes(plain_password),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its stored bcrypt hash."""
    return bcrypt.checkpw(
        _prepare_password_bytes(plain_password),
        hashed_password.encode("utf-8"),
    )


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
