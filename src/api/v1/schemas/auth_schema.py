"""
Request/response DTOs for authentication endpoints.

These are the ONLY objects that cross the HTTP boundary — endpoints never
accept or return ORM models directly.
"""
import re

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, examples=["jane_doe"])
    email: EmailStr = Field(examples=["jane@example.com"])
    password: str = Field(min_length=8, max_length=128, examples=["StrongP@ss123"])

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username may only contain letters, numbers, and underscores.")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v) or not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one letter and one number.")
        return v


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str
