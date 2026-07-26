"""
Integration tests for the authentication and profile endpoints, exercised
through the FastAPI TestClient against an in-memory SQLite database.
"""

from src.core.config import settings

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

    def test_register_long_password_does_not_500(self, client):
        """
        Regression test: bcrypt has a hard 72-BYTE input limit. Versions
        of bcrypt >=4.1 raise ValueError past that limit instead of the
        old silent-truncation behavior passlib used to paper over. A
        password near the schema's max_length=128 chars must still
        register and log in successfully (via explicit truncation in
        core.security), not 500.
        """
        long_password = "Aa1" + ("x" * 125)  # 128 chars, satisfies strength rule
        response = client.post(
            f"{API_PREFIX}/auth/register", json=_register_payload(password=long_password)
        )
        assert response.status_code == 201

        login_response = client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": "jane@example.com", "password": long_password},
        )
        assert login_response.status_code == 200

    def test_register_invalid_username_rejected(self, client):
        response = client.post(
            f"{API_PREFIX}/auth/register", json=_register_payload(username="bad name!")
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
