"""Pydantic schemas for API request and response models."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt


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


class RetrieveRequest(BaseModel):
    """Request payload for semantic retrieval."""

    question: str = Field(
        json_schema_extra={"maxLength": 1000},
    )
    top_k: Optional[StrictInt] = Field(
        default=None,
        json_schema_extra={"minimum": 1, "maximum": 10},
    )


class RetrieveMatch(BaseModel):
    """Individual retrieved chunk match with canonical metadata and raw distance."""

    model_config = ConfigDict(from_attributes=True)

    text: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    page_number: Optional[int] = None
    distance: float


class RetrieveResponse(BaseModel):
    """Response payload for semantic retrieval without generation."""

    model_config = ConfigDict(from_attributes=True)

    question: str
    requested_top_k: int
    has_relevant_context: bool
    retrieved_count: int
    matches: list[RetrieveMatch]
