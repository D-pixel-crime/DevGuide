"""Database and storage adapters for DevGuide."""

from app.db.chroma import (
    DEFAULT_COLLECTION_NAME,
    HNSW_SPACE_COSINE,
    ChromaAdapter,
    ChromaError,
    ChromaReadinessError,
    ChromaStorageError,
    ChromaValidationError,
    ChunkRecord,
    StoredChunkRecord,
    VectorSearchResult,
    check_chroma_readiness,
    format_chunk_id,
)
from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DocumentRepositoryError,
    DuplicateDocumentError,
)
from app.db.sqlite import (
    DOCUMENTS_TABLE_NAME,
    check_sqlite_readiness,
    get_connection,
    init_db,
)

__all__ = [
    # SQLite
    "DOCUMENTS_TABLE_NAME",
    "DocumentNotFoundError",
    "DocumentRepository",
    "DocumentRepositoryError",
    "DuplicateDocumentError",
    "check_sqlite_readiness",
    "get_connection",
    "init_db",
    # ChromaDB
    "DEFAULT_COLLECTION_NAME",
    "HNSW_SPACE_COSINE",
    "ChromaAdapter",
    "ChromaError",
    "ChromaValidationError",
    "ChromaStorageError",
    "ChromaReadinessError",
    "ChunkRecord",
    "VectorSearchResult",
    "StoredChunkRecord",
    "format_chunk_id",
    "check_chroma_readiness",
]
