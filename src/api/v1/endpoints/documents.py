"""
Document management routes.

Every route depends on `get_current_user`, and every service call is
scoped to `current_user.id` — a user can only ever see, check the status
of, or delete their own documents. Thin controllers only: validation and
orchestration live in DocumentService.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from src.api.deps import get_current_user, get_document_service
from src.api.v1.schemas.document_schema import (
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
)
from src.models.user_model import User
from src.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document (PDF, DOCX, or TXT) for RAG ingestion",
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """
    Validates the file type (PDF, DOCX, or TXT), stores it under
    `uploads/`, creates a `documents` row with status PENDING, and
    returns immediately (202 Accepted). Processing is not triggered
    by this endpoint.
    """
    document = await document_service.upload(file, user_id=current_user.id)
    return DocumentResponse.model_validate(document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List the current user's documents",
)
def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents, total = document_service.list_documents(
        user_id=current_user.id, limit=limit, offset=offset
    )
    return DocumentListResponse(
        total=total,
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document's metadata",
)
def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = document_service.get_document(document_id, user_id=current_user.id)
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Poll a document's processing status",
)
def get_document_status(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentStatusResponse:
    document = document_service.get_document(document_id, user_id=current_user.id)
    return DocumentStatusResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and all its vectors",
)
def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> None:
    document_service.delete_document(document_id, user_id=current_user.id)
