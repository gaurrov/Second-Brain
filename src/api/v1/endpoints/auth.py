"""
Authentication routes.

Thin controllers only: validate the request via Pydantic (handled
automatically by FastAPI), delegate to AuthService, and shape the
response. No business logic lives in this file.
"""
from fastapi import APIRouter, Depends, status

from src.api.deps import get_auth_service
from src.api.v1.schemas.auth_schema import (
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from src.api.v1.schemas.user_schema import UserResponse
from src.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(
    payload: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    user = auth_service.register(payload)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate and receive access + refresh tokens",
)
def login(
    payload: UserLoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    user = auth_service.authenticate(payload)
    return auth_service.issue_tokens(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange a valid refresh token for a new token pair",
)
def refresh(
    payload: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return auth_service.refresh_access_token(payload.refresh_token)
