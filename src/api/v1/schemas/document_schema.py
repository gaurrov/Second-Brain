"""
Request/response DTOs for the document module.

`DocumentResponse` is built from the `Document` ORM model. The model's
`created_at` column is exposed to API consumers as `upload_date` via a
validation alias, matching the field name in the required schema without
duplicating the timestamp column at the DB level.
"""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.core.constants import FileType, ProcessingStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    user_id: UUID
    filename: str
    file_type: FileType
    upload_date: datetime = Field(validation_alias="created_at")
    processing_status: ProcessingStatus
    chunk_count: int
    file_size_bytes: int
    error_message: str | None = None
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    processing_status: ProcessingStatus
    chunk_count: int
    error_message: str | None = None
    updated_at: datetime | None = None
