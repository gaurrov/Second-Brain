"""
Response DTOs for user resources.

`UserResponse` uses `from_attributes=True` so it can be constructed
directly from a SQLAlchemy `User` ORM instance (Pydantic v2 style).
"""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime
    updated_at: datetime
