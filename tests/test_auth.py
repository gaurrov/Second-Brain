"""
Integration tests for the authentication and profile endpoints, exercised
through the FastAPI TestClient against an in-memory SQLite database.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from src.core.security import TokenType, create_access_token, decode_token
from src.core.config import settings
from src.models.user_model import User

API_PREFIX = settings.API_V1_PREFIX


def _register_payload(**overrides):
    payload = {
        "username": "jane_doe",
        "email": "jane@example.com",
        "password": "StrongP@ss123",
    }
    payload.update(overrides)
    return payload


class TestRegister:
    def test_register_success(self, client):
        response = client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == "jane@example.com"
        assert body["username"] == "jane_doe"
        assert "hashed_password" not in body  # never leak the hash

    def test_register_duplicate_email_rejected(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        response = client.post(
            f"{API_PREFIX}/auth/register", json=_register_payload(username="another_user")
        )
        assert response.status_code == 409

    def test_register_weak_password_rejected(self, client):
        response = client.post(
            f"{API_PREFIX}/auth/register", json=_register_payload(password="allletters")
        )
        assert response.status_code == 422  # Pydantic validation failure

    def test_register_normalizes_email_and_username(self, client):
        response = client.post(
            f"{API_PREFIX}/auth/register",
            json=_register_payload(username="Jane_Doe", email="Jane@Example.COM"),
        )
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == "jane@example.com"
        assert body["username"] == "jane_doe"
        assert body["role"] == "user"

        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "JANE@EXAMPLE.COM", "password": "StrongP@ss123"},
        )
        assert login_response.status_code == 200

    def test_register_long_password_rejected_without_500(self, client):
        """
        bcrypt has a hard 72-byte input limit. Long passwords must be
        rejected during request validation rather than truncated or allowed
        to raise from the hashing layer as a 500.
        """
        long_password = "Aa1!" + ("x" * 69)  # 73 bytes, satisfies complexity
        response = client.post(
            f"{API_PREFIX}/auth/register", json=_register_payload(password=long_password)
        )
        assert response.status_code == 422

    def test_register_invalid_username_rejected(self, client):
        response = client.post(
            f"{API_PREFIX}/auth/register", json=_register_payload(username="bad name!")
        )
        assert response.status_code == 422

    def test_register_password_cannot_contain_identity(self, client):
        response = client.post(
            f"{API_PREFIX}/auth/register",
            json=_register_payload(password="Jane_doeP@ss123"),
        )
        assert response.status_code == 422


class TestLogin:
    def test_login_success_returns_token_pair(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_login_wrong_password_rejected(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "WrongPassword1"},
        )
        assert response.status_code == 401

    def test_login_unknown_email_rejected(self, client):
        response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "ghost@example.com", "password": "WhoKnows123"},
        )
        assert response.status_code == 401

    def test_login_failure_tracks_lock_state_without_changing_error(self, client, db_session):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())

        for _ in range(settings.MAX_FAILED_LOGIN_ATTEMPTS):
            response = client.post(
                f"{API_PREFIX}/auth/login",
                json={"email": "jane@example.com", "password": "WrongPassword1!"},
            )
            assert response.status_code == 401
            assert response.json()["detail"] == "Incorrect email or password."

        user = db_session.execute(
            select(User).where(User.email == "jane@example.com")
        ).scalar_one()
        assert user.failed_login_attempts == settings.MAX_FAILED_LOGIN_ATTEMPTS
        assert user.locked_until is not None

        locked_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        assert locked_response.status_code == 401
        assert locked_response.json()["detail"] == "Incorrect email or password."

    def test_successful_login_resets_failed_attempts(self, client, db_session):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "WrongPassword1!"},
        )

        response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        assert response.status_code == 200

        user = db_session.execute(
            select(User).where(User.email == "jane@example.com")
        ).scalar_one()
        assert user.failed_login_attempts == 0
        assert user.locked_until is None
        assert user.last_login_at is not None


class TestProfile:
    def test_profile_requires_authentication(self, client):
        response = client.get(f"{API_PREFIX}/users/profile")
        assert response.status_code == 401

    def test_profile_returns_current_user(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        access_token = login_response.json()["access_token"]

        response = client.get(
            f"{API_PREFIX}/users/profile",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        assert response.json()["email"] == "jane@example.com"

    def test_profile_rejects_garbage_token(self, client):
        response = client.get(
            f"{API_PREFIX}/users/profile",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 401

    def test_profile_rejects_expired_access_token(self, client, monkeypatch):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        user_id = client.get(
            f"{API_PREFIX}/users/profile",
            headers={
                "Authorization": "Bearer "
                + client.post(
                    f"{API_PREFIX}/auth/login",
                    json={"email": "jane@example.com", "password": "StrongP@ss123"},
                ).json()["access_token"]
            },
        ).json()["id"]

        monkeypatch.setattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", -1)
        expired_token = create_access_token(user_id)

        response = client.get(
            f"{API_PREFIX}/users/profile",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401


class TestRefreshToken:
    def test_refresh_issues_new_token_pair(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        refresh_token = login_response.json()["refresh_token"]

        response = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": refresh_token})
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert response.json()["refresh_token"] != refresh_token

    def test_refresh_rejects_access_token_used_as_refresh(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        access_token = login_response.json()["access_token"]

        # Using an access token where a refresh token is expected must fail.
        response = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": access_token})
        assert response.status_code == 401

    def test_refresh_rotation_rejects_replayed_refresh_token(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        original_refresh_token = login_response.json()["refresh_token"]

        refresh_response = client.post(
            f"{API_PREFIX}/auth/refresh",
            json={"refresh_token": original_refresh_token},
        )
        assert refresh_response.status_code == 200

        replay_response = client.post(
            f"{API_PREFIX}/auth/refresh",
            json={"refresh_token": original_refresh_token},
        )
        assert replay_response.status_code == 401

        rotated_response = client.post(
            f"{API_PREFIX}/auth/refresh",
            json={"refresh_token": refresh_response.json()["refresh_token"]},
        )
        assert rotated_response.status_code == 200

    def test_refresh_token_contains_required_expiration_and_id(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        payload = decode_token(login_response.json()["refresh_token"], TokenType.REFRESH)

        assert payload["jti"]
        assert payload["exp"] > int(datetime.now(timezone.utc).timestamp())


class TestLogout:
    def test_logout_requires_authentication(self, client):
        response = client.post(f"{API_PREFIX}/auth/logout")
        assert response.status_code == 401

    def test_logout_returns_204(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        access_token = login_response.json()["access_token"]

        response = client.post(
            f"{API_PREFIX}/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 204

    def test_logout_revokes_refresh_token(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        refresh_token = login_response.json()["refresh_token"]
        access_token = login_response.json()["access_token"]

        # Logout — revokes all refresh tokens.
        client.post(
            f"{API_PREFIX}/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # The old refresh token must no longer work.
        response = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": refresh_token})
        assert response.status_code == 401

    def test_logout_does_not_invalidate_access_token_immediately(self, client):
        """
        JWTs are stateless — logout only revokes refresh tokens.
        The access token stays valid until it naturally expires.
        """
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        access_token = login_response.json()["access_token"]

        client.post(
            f"{API_PREFIX}/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # The access token is still a valid JWT — profile still works.
        response = client.get(
            f"{API_PREFIX}/users/profile",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200

    def test_logout_twice_is_idempotent(self, client):
        client.post(f"{API_PREFIX}/auth/register", json=_register_payload())
        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": "StrongP@ss123"},
        )
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        assert client.post(f"{API_PREFIX}/auth/logout", headers=headers).status_code == 204
        assert client.post(f"{API_PREFIX}/auth/logout", headers=headers).status_code == 204
