"""Application service layer."""

from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    InvalidEmbeddingResponseError,
    MissingApiKeyError,
    OpenAIEmbeddingAdapter,
)
from app.services.ingestion import (
    ChunkingError,
    CorruptPDFError,
    EmptyExtractionError,
    EmptyFileError,
    FileContentMismatchError,
    FileSizeLimitExceededError,
    FileValidationError,
    ImageOnlyPDFError,
    IngestionError,
    InvalidEncodingError,
    InvalidFilenameError,
    TextExtractionError,
    UnsupportedFileTypeError,
    chunk_document,
    chunk_text,
    extract_text,
    validate_filename,
    validate_upload,
)

__all__ = [
    # Ingestion
    "IngestionError",
    "FileValidationError",
    "UnsupportedFileTypeError",
    "FileSizeLimitExceededError",
    "EmptyFileError",
    "InvalidFilenameError",
    "FileContentMismatchError",
    "InvalidEncodingError",
    "TextExtractionError",
    "CorruptPDFError",
    "ImageOnlyPDFError",
    "EmptyExtractionError",
    "ChunkingError",
    "validate_filename",
    "validate_upload",
    "extract_text",
    "chunk_text",
    "chunk_document",
    # Embedding
    "EmbeddingError",
    "MissingApiKeyError",
    "EmbeddingAuthenticationError",
    "EmbeddingProviderError",
    "InvalidEmbeddingResponseError",
    "OpenAIEmbeddingAdapter",
]
