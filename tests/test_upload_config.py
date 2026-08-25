"""
Tests for the upload-configuration endpoint (GET /api/v1/config/upload).

Verifies that the endpoint exposes the live backend configuration values
(max upload size in MB and the allowed-extension allow-list) so clients
can pre-validate files before transferring them.
"""
from src.core.config import settings
from src.core.constants import ALLOWED_EXTENSIONS

API_PREFIX = settings.API_V1_PREFIX


class TestUploadConfig:
    def test_returns_configured_max_size_and_extensions(self, client):
        response = client.get(f"{API_PREFIX}/config/upload")
        assert response.status_code == 200
        body = response.json()
        assert body["max_upload_size_mb"] == settings.MAX_UPLOAD_SIZE_MB
        assert body["allowed_extensions"] == sorted(ALLOWED_EXTENSIONS.keys())
        # The formats advertised by the product's primary upload flow.
        for extension in (".pdf", ".docx", ".txt"):
            assert extension in body["allowed_extensions"]

    def test_requires_no_authentication(self, client):
        response = client.get(f"{API_PREFIX}/config/upload")
        assert response.status_code == 200

    def test_reflects_configuration_change(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 42)
        response = client.get(f"{API_PREFIX}/config/upload")
        assert response.status_code == 200
        assert response.json()["max_upload_size_mb"] == 42
