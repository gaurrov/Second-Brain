"""
Security primitives: password hashing (bcrypt) and JWT
creation/validation (python-jose). No business logic lives here — this
module only deals with cryptographic concerns.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import bcrypt
from jose import JWTError, jwt

from src.core.config import settings
from src.core.exceptions import InvalidTokenException, TokenExpiredException

# bcrypt operates on bytes and has a hard 72-BYTE input limit (not 72
# characters — a multi-byte UTF-8 password can exceed this well under
# 72 characters). bcrypt>=4.1 raises ValueError past this limit instead
# of silently truncating like older versions / passlib's CryptContext
# did. We reject over-limit passwords during validation so hashing and
# verification never silently treat different long inputs as equivalent.
_BCRYPT_MAX_BYTES = 72
_DUMMY_PASSWORD_HASH = bcrypt.hashpw(
    b"not-a-real-user-password",
    bcrypt.gensalt(),
).decode("utf-8")


def _prepare_password_bytes(plain_password: str) -> bytes:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > _BCRYPT_MAX_BYTES:
        raise ValueError("Password exceeds bcrypt's 72-byte limit.")
    return password_bytes


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
    try:
        password_bytes = _prepare_password_bytes(plain_password)
    except ValueError:
        return False
    return bcrypt.checkpw(
        password_bytes,
        hashed_password.encode("utf-8"),
    )


def verify_password_for_unknown_user(plain_password: str) -> None:
    """Run a dummy bcrypt check to reduce obvious login timing differences."""
    verify_password(plain_password, _DUMMY_PASSWORD_HASH)


# --------------------------------------------------------------------------
# JWT creation
# --------------------------------------------------------------------------
def access_token_lifetime() -> timedelta:
    return timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)


def refresh_token_lifetime() -> timedelta:
    return timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)


def hash_token_identifier(token_id: str | UUID) -> str:
    return hashlib.sha256(str(token_id).encode("utf-8")).hexdigest()


def _create_token(
    subject: str | UUID,
    token_type: TokenType,
    expires_delta: timedelta,
    token_id: str | UUID | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type.value,
        "jti": str(token_id or uuid4()),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: str | UUID) -> str:
    """Create a short-lived JWT access token carrying the user's id as `sub`."""
    return _create_token(
        subject=user_id,
        token_type=TokenType.ACCESS,
        expires_delta=access_token_lifetime(),
    )


def create_refresh_token(user_id: str | UUID, token_id: str | UUID | None = None) -> str:
    """Create a long-lived JWT refresh token."""
    return _create_token(
        subject=user_id,
        token_type=TokenType.REFRESH,
        expires_delta=refresh_token_lifetime(),
        token_id=token_id,
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
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require_exp": True, "require_iat": True, "require_sub": True},
        )
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException()
    except JWTError:
        raise InvalidTokenException()

    if payload.get("type") != expected_type.value:
        raise InvalidTokenException(f"Expected a {expected_type.value} token.")

    if not payload.get("sub"):
        raise InvalidTokenException()

    if expected_type is TokenType.REFRESH and not payload.get("jti"):
        raise InvalidTokenException()

    return payload
