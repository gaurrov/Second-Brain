"""
Upload-configuration endpoint.

Exposes the upload constraints (max file size, accepted extensions) from
the single source of truth (`src/core/config.py` / `src/core/constants.py`)
so API consumers can pre-validate files client-side BEFORE transferring
them. Nothing here is sensitive, hence no authentication requirement.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from src.core.config import settings
from src.core.constants import ALLOWED_EXTENSIONS

router = APIRouter(prefix="/config", tags=["Configuration"])


class UploadConfigResponse(BaseModel):
    max_upload_size_mb: int
    allowed_extensions: list[str]


@router.get(
    "/upload",
    response_model=UploadConfigResponse,
    summary="Get upload constraints (max size in MB, allowed extensions)",
)
def get_upload_config() -> UploadConfigResponse:
    return UploadConfigResponse(
        max_upload_size_mb=settings.MAX_UPLOAD_SIZE_MB,
        allowed_extensions=sorted(ALLOWED_EXTENSIONS.keys()),
    )
