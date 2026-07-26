"""
Request/response DTOs for authentication endpoints.

These are the ONLY objects that cross the HTTP boundary — endpoints never
accept or return ORM models directly.
"""
import re

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class UserRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, examples=["jane_doe"])
    email: EmailStr = Field(examples=["jane@example.com"])
    password: str = Field(min_length=12, max_length=128, examples=["StrongP@ss123!"])

    @field_validator("username")
    @classmethod
    def normalize_username(cls, v: str) -> str:
        username = v.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,49}", username):
            raise ValueError(
                "Username must start with a letter and contain only lowercase letters, "
                "numbers, and underscores."
            )
        if "__" in username or username.endswith("_"):
            raise ValueError("Username cannot contain consecutive or trailing underscores.")
        return username

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> str:
        return str(v).strip().lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer.")
        if any(char.isspace() for char in v):
            raise ValueError("Password cannot contain whitespace.")
        checks = (
            re.search(r"[a-z]", v),
            re.search(r"[A-Z]", v),
            re.search(r"[0-9]", v),
            re.search(r"[^A-Za-z0-9]", v),
        )
        if not all(checks):
            raise ValueError(
                "Password must include lowercase, uppercase, number, and special characters."
            )
        return v

    @model_validator(mode="after")
    def password_must_not_contain_identity(self) -> "UserRegisterRequest":
        password_lower = self.password.lower()
        email_local = str(self.email).split("@", 1)[0].lower()
        if self.username in password_lower or email_local in password_lower:
            raise ValueError("Password cannot contain your username or email name.")
        return self


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> str:
        return str(v).strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str
