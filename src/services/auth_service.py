"""
Authentication service — the use-case layer for registration, login, and
token refresh. Endpoints call into this; this is where business rules
live (uniqueness checks, credential verification, token issuance). It
knows nothing about HTTP — it raises domain exceptions from
src.core.exceptions, which the API layer translates to HTTP responses.
"""
import logging

from datetime import datetime, timezone
from uuid import UUID
from uuid import uuid4

from src.api.v1.schemas.auth_schema import TokenResponse, UserLoginRequest, UserRegisterRequest
from src.core.config import settings
from src.core.exceptions import (
    InactiveUserException,
    InvalidCredentialsException,
    InvalidTokenException,
    UserAlreadyExistsException,
)
from src.core.security import (
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    refresh_token_lifetime,
    verify_password,
    verify_password_for_unknown_user,
)
from src.models.user_model import User
from src.repositories.refresh_token_repository import RefreshTokenRepository
from src.repositories.user_repository import UserRepository

logger = logging.getLogger("second_brain.auth_service")


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        refresh_token_repository: RefreshTokenRepository,
    ) -> None:
        self.user_repository = user_repository
        self.refresh_token_repository = refresh_token_repository

    def register(self, payload: UserRegisterRequest) -> User:
        """Create a new user account. Raises UserAlreadyExistsException on conflict."""
        if self.user_repository.exists_by_email_or_username(
            email=payload.email, username=payload.username
        ):
            raise UserAlreadyExistsException()

        user = User(
            username=payload.username,
            email=payload.email,
            hashed_password=hash_password(payload.password),
        )
        user = self.user_repository.create(user)
        logger.info("User registered: id=%s username=%s", user.id, user.username)
        return user

    def authenticate(self, payload: UserLoginRequest) -> User:
        """
        Validate email/password. Deliberately raises the SAME exception
        for "user not found" and "wrong password" to avoid leaking which
        emails are registered (user enumeration protection).
        """
        user = self.user_repository.get_by_email(payload.email)
        if user is None:
            verify_password_for_unknown_user(payload.password)
            raise InvalidCredentialsException()

        if self._is_locked(user) or not verify_password(payload.password, user.hashed_password):
            self.user_repository.record_failed_login(
                user=user,
                max_attempts=settings.MAX_FAILED_LOGIN_ATTEMPTS,
                lock_minutes=settings.ACCOUNT_LOCK_MINUTES,
            )
            raise InvalidCredentialsException()

        if not user.is_active:
            raise InactiveUserException()

        self.user_repository.reset_login_security_state(user)
        logger.info("User authenticated: id=%s", user.id)
        return user

    def issue_tokens(self, user: User) -> TokenResponse:
        refresh_token_id = uuid4()
        refresh_token = create_refresh_token(user.id, token_id=refresh_token_id)
        self.refresh_token_repository.add_for_user(
            user_id=user.id,
            token_id=refresh_token_id,
            expires_at=datetime.now(timezone.utc) + refresh_token_lifetime(),
        )
        self.refresh_token_repository.db.commit()
        return TokenResponse(
            access_token=create_access_token(user.id),
            refresh_token=refresh_token,
        )

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """
        Validate a refresh token and issue a brand new access + refresh
        token pair (refresh-token rotation).
        """
        payload = decode_token(refresh_token, expected_type=TokenType.REFRESH)
        persisted_token = self.refresh_token_repository.get_active_by_token_id(payload["jti"])
        if persisted_token is None:
            raise InvalidTokenException()

        try:
            user_id = UUID(payload["sub"])
        except (ValueError, TypeError):
            raise InvalidTokenException()

        user = self.user_repository.get_by_id(user_id)

        if user is None:
            raise InvalidTokenException()
        if not user.is_active:
            raise InactiveUserException()

        new_refresh_token_id = uuid4()
        new_refresh_token = create_refresh_token(user.id, token_id=new_refresh_token_id)
        self.refresh_token_repository.add_for_user(
            user_id=user.id,
            token_id=new_refresh_token_id,
            expires_at=datetime.now(timezone.utc) + refresh_token_lifetime(),
        )
        self.refresh_token_repository.revoke(
            persisted_token,
            replaced_by_token_id=new_refresh_token_id,
        )
        self.refresh_token_repository.db.commit()

        logger.info("Tokens refreshed for user: id=%s", user.id)
        return TokenResponse(
            access_token=create_access_token(user.id),
            refresh_token=new_refresh_token,
        )

    def _is_locked(self, user: User) -> bool:
        if user.locked_until is None:
            return False
        locked_until = user.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        return locked_until > datetime.now(timezone.utc)
