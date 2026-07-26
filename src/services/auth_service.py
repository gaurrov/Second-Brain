"""
Authentication service — the use-case layer for registration, login, and
token refresh. Endpoints call into this; this is where business rules
live (uniqueness checks, credential verification, token issuance). It
knows nothing about HTTP — it raises domain exceptions from
src.core.exceptions, which the API layer translates to HTTP responses.
"""
import logging

from uuid import UUID

from src.api.v1.schemas.auth_schema import TokenResponse, UserLoginRequest, UserRegisterRequest
from src.core.exceptions import (
    InactiveUserException,
    InvalidCredentialsException,
    UserAlreadyExistsException,
    UserNotFoundException,
)
from src.core.security import (
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.models.user_model import User
from src.repositories.user_repository import UserRepository

logger = logging.getLogger("second_brain.auth_service")


class AuthService:
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository

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
        if user is None or not verify_password(payload.password, user.hashed_password):
            raise InvalidCredentialsException()

        if not user.is_active:
            raise InactiveUserException()

        logger.info("User authenticated: id=%s", user.id)
        return user

    def issue_tokens(self, user: User) -> TokenResponse:
        return TokenResponse(
            access_token=create_access_token(user.id),
            refresh_token=create_refresh_token(user.id),
        )

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """
        Validate a refresh token and issue a brand new access + refresh
        token pair (refresh-token rotation).
        """
        payload = decode_token(refresh_token, expected_type=TokenType.REFRESH)
        user = self.user_repository.get_by_id(UUID(payload["sub"]))

        if user is None:
            raise UserNotFoundException()
        if not user.is_active:
            raise InactiveUserException()

        logger.info("Tokens refreshed for user: id=%s", user.id)
        return self.issue_tokens(user)
