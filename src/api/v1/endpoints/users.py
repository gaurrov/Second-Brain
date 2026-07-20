"""
User routes. `/profile` is a protected route — it demonstrates the
standard pattern for guarding any endpoint: depend on `get_current_user`
and FastAPI + the dependency chain handle the rest (missing token ->
401, expired token -> 401, inactive user -> 403).
"""
from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.api.v1.schemas.user_schema import UserResponse
from src.models.user_model import User

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/profile",
    response_model=UserResponse,
    summary="Get the currently authenticated user's profile",
)
def get_profile(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)
