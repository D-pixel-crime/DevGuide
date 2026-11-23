"""Pydantic schemas for API request and response models."""

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Document metadata response matching API specification."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    content_sha256: str
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    """Response containing list of documents and total count."""

    documents: list[DocumentResponse]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response returned upon successful document deletion."""

    id: str
    deleted: bool
    chunks_removed: int


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str
