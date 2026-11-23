"""REST API routes for document upload, listing, detail, and deletion."""

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.chroma import ChromaAdapter, ChromaError
from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DuplicateDocumentError,
)
from app.models.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
)
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    OpenAIEmbeddingAdapter,
)
from app.services.ingestion import (
    ChromaDeletionError,
    ChunkingError,
    DeletionPartialConsistencyError,
    FileSizeLimitExceededError,
    FileValidationError,
    IngestionService,
    IngestionStorageError,
    PartialConsistencyError,
    TextExtractionError,
)

logger = get_logger("app.api.routes_documents")

router = APIRouter(tags=["documents"])

CHUNK_READ_SIZE = 64 * 1024  # 64 KB read buffer for streaming size enforcement


def get_app_settings(request: Request) -> Settings:
    """Retrieve application-scoped Settings from app state with fallback to get_settings()."""
    app_settings = getattr(request.app.state, "settings", None)
    if app_settings is not None and isinstance(app_settings, Settings):
        return app_settings
    return get_settings()


def read_upload_file_bytes(upload_file: UploadFile, max_bytes: int) -> bytes:
    """Read bytes from UploadFile enforcing max_bytes limit while streaming.

    Reads in chunks and raises FileSizeLimitExceededError immediately if cumulative
    byte count exceeds max_bytes, without loading excess bytes into memory.
    """
    limit_mb = max_bytes // (1024 * 1024)

    # Early fast-reject if Content-Length or file.size is known and exceeds limit
    if upload_file.size is not None and upload_file.size > max_bytes:
        raise FileSizeLimitExceededError(
            f"File size exceeds maximum allowed upload limit of {limit_mb} MB ({max_bytes} bytes)"
        )

    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        chunk = upload_file.file.read(CHUNK_READ_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise FileSizeLimitExceededError(
                f"File size exceeds maximum allowed upload limit of {limit_mb} MB ({max_bytes} bytes)"
            )
        chunks.append(chunk)

    return b"".join(chunks)


def get_ingestion_service(
    settings: Settings = Depends(get_app_settings),
) -> IngestionService:
    """Dependency provider for IngestionService using application-scoped settings."""
    repository = DocumentRepository(db_path=settings.SQLITE_PATH)
    chroma_adapter = ChromaAdapter(chroma_path=settings.CHROMA_PATH, settings=settings)
    embedding_adapter = OpenAIEmbeddingAdapter(settings=settings)
    return IngestionService(
        repository=repository,
        chroma_adapter=chroma_adapter,
        embedding_adapter=embedding_adapter,
        settings=settings,
    )


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file type, corrupt file, or empty content"},
        409: {"model": ErrorResponse, "description": "Duplicate document content SHA-256 hash already indexed"},
        413: {"model": ErrorResponse, "description": "Uploaded file exceeds maximum allowed upload limit"},
        500: {"model": ErrorResponse, "description": "Storage failure, partial consistency failure, or internal server error"},
        503: {"model": ErrorResponse, "description": "Upstream OpenAI API failure during embedding generation"},
    },
)
def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_app_settings),
) -> DocumentResponse:
    """Upload and index a document (.pdf, .md, or .txt)."""
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    try:
        content = read_upload_file_bytes(file, max_bytes=max_bytes)
        filename = file.filename or ""
        document = service.ingest_document(filename=filename, content=content)
        return DocumentResponse.model_validate(document)
    except FileSizeLimitExceededError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=str(exc),
        ) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (FileValidationError, TextExtractionError, ChunkingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PartialConsistencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except IngestionStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except (EmbeddingProviderError, EmbeddingAuthenticationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (EmbeddingError, ChromaError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage or embedding service failure during ingestion",
        ) from exc


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all indexed documents",
    responses={
        500: {"model": ErrorResponse, "description": "Internal storage failure during document listing"},
    },
)
def list_documents(
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    """List all indexed documents and their summary metadata."""
    docs = service.list_documents()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve document metadata by ID",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid document UUID format"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal storage failure during document retrieval"},
    },
)
def get_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentResponse:
    """Retrieve metadata for a single document by its UUID."""
    try:
        doc = service.get_document(document_id)
        return DocumentResponse.model_validate(doc)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete document and its indexed vectors",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid document UUID format"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Chroma deletion failure or partial consistency failure"},
    },
)
def delete_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentDeleteResponse:
    """Delete a document from SQLite and delete all corresponding chunks from ChromaDB."""
    try:
        result = service.delete_document(document_id)
        return DocumentDeleteResponse(
            id=result.id,
            deleted=result.deleted,
            chunks_removed=result.chunks_removed,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ChromaDeletionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except DeletionPartialConsistencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

