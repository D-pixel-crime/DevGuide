"""Persistent ChromaDB vector-store adapter with cosine space and explicit embeddings."""

import math
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk

logger = get_logger("app.db.chroma")

DEFAULT_COLLECTION_NAME: str = "devguide_chunks"
HNSW_SPACE_COSINE: str = "cosine"


class ChromaError(Exception):
    """Base exception for ChromaDB vector-store adapter errors."""


class ChromaValidationError(ChromaError, ValueError):
    """Raised when chunk metadata, vector dimension, or ID cardinality validation fails."""


class ChromaStorageError(ChromaError):
    """Raised when an internal ChromaDB storage, write, or query operation fails."""


class ChromaReadinessError(ChromaError):
    """Raised when ChromaDB readiness probe fails."""


def format_chunk_id(document_id: str, chunk_index: int) -> str:
    """Format canonical deterministic chunk ID (<document_uuid>:<chunk_index>)."""
    return f"{document_id}:{chunk_index}"


@dataclass(frozen=True)
class ChunkRecord:
    """Domain model for a document chunk to be stored in ChromaDB.

    Attributes:
        document_id: UUID string of the parent document.
        chunk_index: Sequential 0-based integer across the document.
        text: Non-empty chunk text content.
        embedding: Dense float embedding vector.
        filename: Safe filename of the parent document.
        file_type: File extension ('pdf', 'md', 'txt').
        page_number: 1-based page number for PDF chunks; None for Markdown/TXT.
        id: Optional explicit chunk ID (defaults to format_chunk_id).
    """

    document_id: str
    chunk_index: int
    text: str
    embedding: list[float]
    filename: str
    file_type: str
    page_number: Optional[int] = None
    id: Optional[str] = None

    @property
    def chunk_id(self) -> str:
        """Deterministic chunk ID."""
        if self.id is not None:
            return self.id
        return format_chunk_id(self.document_id, self.chunk_index)

    @classmethod
    def from_document_chunk(
        cls,
        chunk: DocumentChunk,
        embedding: list[float],
        filename: str,
        file_type: str,
    ) -> "ChunkRecord":
        """Construct a ChunkRecord from a DocumentChunk and an embedding vector."""
        if chunk.document_id is None:
            raise ChromaValidationError(
                "Cannot construct ChunkRecord from DocumentChunk without document_id"
            )
        return cls(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=embedding,
            filename=filename,
            file_type=file_type,
            page_number=chunk.page_number,
        )


@dataclass(frozen=True)
class VectorSearchResult:
    """Result item from a vector similarity search.

    Attributes:
        id: Chunk ID (<document_uuid>:<chunk_index>).
        document_id: UUID string of the parent document.
        chunk_index: Sequential 0-based integer across the document.
        text: Raw chunk text.
        filename: Safe filename.
        file_type: Normalized file type.
        page_number: 1-based page number for PDF chunks; None for Markdown/TXT.
        distance: Geometric cosine distance returned by ChromaDB (0.0 to 2.0).
    """

    id: str
    document_id: str
    chunk_index: int
    text: str
    filename: str
    file_type: str
    page_number: Optional[int]
    distance: float


@dataclass(frozen=True)
class StoredChunkRecord:
    """Stored chunk record retrieved from ChromaDB.

    Attributes:
        id: Chunk ID.
        document_id: UUID string of the parent document.
        chunk_index: Sequential 0-based integer across the document.
        text: Raw chunk text.
        filename: Safe filename.
        file_type: Normalized file type.
        page_number: 1-based page number for PDF chunks; None for Markdown/TXT.
        embedding: Optional embedding vector if requested.
    """

    id: str
    document_id: str
    chunk_index: int
    text: str
    filename: str
    file_type: str
    page_number: Optional[int]
    embedding: Optional[list[float]] = None


def check_chroma_readiness(
    chroma_path: Optional[Union[pathlib.Path, str]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> str:
    """Verify local ChromaDB availability without creating missing stores as a side effect.

    Args:
        chroma_path: Path to ChromaDB persistent directory (defaults to Settings.CHROMA_PATH).
        collection_name: Name of the expected collection.

    Returns:
        'ok': If persistent storage exists and collection is accessible.
        'not_initialized': If storage path or collection does not exist on disk.
        'error': If storage is corrupted or connection/query fails.
    """
    if chroma_path is None:
        chroma_path = get_settings().CHROMA_PATH

    target_path = pathlib.Path(chroma_path)

    # Must not create missing store or directories as a side effect
    if not target_path.exists():
        return "not_initialized"

    if not target_path.is_dir():
        logger.error("ChromaDB target path %s is not a directory", target_path)
        return "error"

    chroma_db_file = target_path / "chroma.sqlite3"
    if not chroma_db_file.exists():
        return "not_initialized"

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(target_path))
        client.heartbeat()

        try:
            client.get_collection(name=collection_name, embedding_function=None)
            return "ok"
        except Exception as col_exc:
            exc_name = type(col_exc).__name__
            if "NotFoundError" in exc_name or "InvalidCollection" in exc_name:
                return "not_initialized"
            logger.error("Failed to access collection %s: %s", collection_name, exc_name)
            return "error"
    except Exception as exc:
        logger.error("ChromaDB readiness check failed: %s", type(exc).__name__)
        return "error"


class ChromaAdapter:
    """Adapter for persistent local ChromaDB vector store.

    Enforces cosine distance, passes explicit vectors without silent model selection,
    handles deterministic chunk IDs, and validates cardinality and metadata invariants.
    """

    def __init__(
        self,
        chroma_path: Optional[Union[pathlib.Path, str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize ChromaAdapter lazily without creating files on disk in __init__.

        Args:
            chroma_path: Path to ChromaDB directory (defaults to Settings.CHROMA_PATH).
            collection_name: Name of target collection (defaults to 'devguide_chunks').
            client: Optional injected Chroma client (for testing).
            settings: Optional Settings instance.
        """
        self._chroma_path = chroma_path
        self._collection_name = collection_name
        self._client = client
        self._settings = settings
        self._collection: Optional[Any] = None

    @property
    def chroma_path(self) -> pathlib.Path:
        """Resolved path to ChromaDB persistent storage."""
        if self._chroma_path is not None:
            return pathlib.Path(self._chroma_path)
        settings = self._settings if self._settings is not None else get_settings()
        return settings.CHROMA_PATH

    @property
    def collection_name(self) -> str:
        """Collection name."""
        return self._collection_name

    def _get_client(self) -> Any:
        """Get or lazily initialize persistent ChromaDB client."""
        if self._client is not None:
            return self._client

        target_path = self.chroma_path
        target_path.mkdir(parents=True, exist_ok=True)

        import chromadb

        self._client = chromadb.PersistentClient(path=str(target_path))
        return self._client

    def _get_collection(self) -> Any:
        """Get or create collection with explicit cosine distance and no embedding function."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        try:
            # Configure cosine distance explicitly and disable silent embedding function selection
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": HNSW_SPACE_COSINE},
                embedding_function=None,
            )
            return self._collection
        except Exception as exc:
            logger.error("Failed to get or create ChromaDB collection: %s", type(exc).__name__)
            raise ChromaStorageError(
                f"Failed to access ChromaDB collection '{self.collection_name}': {type(exc).__name__}"
            ) from exc

    def init_collection(self) -> None:
        """Explicitly ensure the persistent client and collection are initialized."""
        _ = self._get_collection()

    def add_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        """Add new chunks to the collection with cardinality and metadata validation.

        Args:
            chunks: Sequence of ChunkRecord items with explicit embeddings.

        Raises:
            ChromaValidationError: If chunks violate schema, vector, or cardinality rules.
            ChromaStorageError: If ChromaDB insertion fails.
        """
        self._write_chunks(chunks, upsert=False)

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        """Upsert chunks into the collection (insert or update).

        Args:
            chunks: Sequence of ChunkRecord items with explicit embeddings.

        Raises:
            ChromaValidationError: If chunks violate schema, vector, or cardinality rules.
            ChromaStorageError: If ChromaDB upsert fails.
        """
        self._write_chunks(chunks, upsert=True)

    def _write_chunks(self, chunks: Sequence[ChunkRecord], upsert: bool = False) -> None:
        """Validate and write chunks to ChromaDB."""
        if not chunks:
            return

        expected_dim: Optional[int] = None
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkRecord):
                raise ChromaValidationError(
                    f"Item at index {idx} must be a ChunkRecord, got {type(chunk).__name__}"
                )

            # Validate document_id
            if not chunk.document_id or not isinstance(chunk.document_id, str) or not chunk.document_id.strip():
                raise ChromaValidationError(
                    f"Chunk at index {idx} has invalid document_id: must be non-empty string"
                )

            # Validate chunk_index
            if not isinstance(chunk.chunk_index, int) or chunk.chunk_index < 0:
                raise ChromaValidationError(
                    f"Chunk at index {idx} has invalid chunk_index: must be non-negative integer, got {chunk.chunk_index}"
                )

            # Validate text
            if not chunk.text or not isinstance(chunk.text, str) or not chunk.text.strip():
                raise ChromaValidationError(
                    f"Chunk at index {idx} has invalid text: must be non-empty string"
                )

            # Validate filename and file_type
            if not chunk.filename or not isinstance(chunk.filename, str):
                raise ChromaValidationError(
                    f"Chunk at index {idx} has invalid filename: must be non-empty string"
                )
            if not chunk.file_type or not isinstance(chunk.file_type, str):
                raise ChromaValidationError(
                    f"Chunk at index {idx} has invalid file_type: must be non-empty string"
                )

            # Validate page_number
            if chunk.page_number is not None:
                if not isinstance(chunk.page_number, int) or chunk.page_number < 1:
                    raise ChromaValidationError(
                        f"Chunk at index {idx} has invalid page_number: must be positive int or None, got {chunk.page_number}"
                    )

            # Validate embedding vector
            if chunk.embedding is None or not isinstance(chunk.embedding, (list, tuple)) or len(chunk.embedding) == 0:
                raise ChromaValidationError(
                    f"Chunk at index {idx} must have a non-empty embedding vector"
                )

            if expected_dim is None:
                expected_dim = len(chunk.embedding)
            elif len(chunk.embedding) != expected_dim:
                raise ChromaValidationError(
                    f"Chunk at index {idx} has vector dimension {len(chunk.embedding)}, expected {expected_dim}"
                )

            # Validate all elements are finite numbers
            float_vector: list[float] = []
            for elem_idx, val in enumerate(chunk.embedding):
                if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                    raise ChromaValidationError(
                        f"Chunk at index {idx}, vector element {elem_idx} is not a finite number"
                    )
                float_vector.append(float(val))

            # Deterministic chunk ID
            derived_id = format_chunk_id(chunk.document_id, chunk.chunk_index)
            if chunk.id is not None and chunk.id != derived_id:
                raise ChromaValidationError(
                    f"Chunk at index {idx} has mismatched explicit id '{chunk.id}', expected '{derived_id}'"
                )

            # Metadata construction: omit page_number for non-PDF / None
            meta: dict[str, Any] = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "file_type": chunk.file_type,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page_number is not None:
                meta["page_number"] = int(chunk.page_number)

            ids.append(derived_id)
            documents.append(chunk.text)
            embeddings.append(float_vector)
            metadatas.append(meta)

        # Cardinality verification
        total_chunks = len(chunks)
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas) == total_chunks):
            raise ChromaValidationError(
                f"Cardinality mismatch before ChromaDB write: ids={len(ids)}, docs={len(documents)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}, chunks={total_chunks}"
            )

        # Duplicate ID check within write batch
        if len(set(ids)) != total_chunks:
            raise ChromaValidationError("Duplicate chunk IDs detected in write batch")

        col = self._get_collection()
        try:
            if upsert:
                col.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            else:
                col.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            logger.info(
                "%s %d chunks in ChromaDB collection '%s'",
                "Upserted" if upsert else "Added",
                total_chunks,
                self.collection_name,
            )
        except Exception as exc:
            logger.error("ChromaDB %s write failed: %s", "upsert" if upsert else "add", type(exc).__name__)
            raise ChromaStorageError(
                f"Failed to write chunks to ChromaDB ({'upsert' if upsert else 'add'}): {type(exc).__name__}"
            ) from exc

    def query_nearest(
        self,
        query_vector: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """Query the collection for the nearest chunks using cosine distance.

        Args:
            query_vector: Dense float embedding vector for the query.
            top_k: Maximum number of nearest neighbors to retrieve.
            where: Optional metadata filter dict.

        Returns:
            List of VectorSearchResult items ordered by cosine distance ascending.

        Raises:
            ChromaValidationError: If query_vector or top_k is invalid.
            ChromaStorageError: If ChromaDB query fails.
        """
        if query_vector is None or not isinstance(query_vector, (list, tuple)) or len(query_vector) == 0:
            raise ChromaValidationError("query_vector must be a non-empty sequence of numbers")

        for elem_idx, val in enumerate(query_vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise ChromaValidationError(
                    f"query_vector element {elem_idx} is not a finite number"
                )

        if not isinstance(top_k, int) or top_k <= 0:
            raise ChromaValidationError(f"top_k must be a positive integer, got {top_k}")

        col = self._get_collection()
        kwargs: dict[str, Any] = {
            "query_embeddings": [[float(x) for x in query_vector]],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            res = col.query(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", type(exc).__name__)
            raise ChromaStorageError(f"ChromaDB nearest neighbor query failed: {type(exc).__name__}") from exc

        results: list[VectorSearchResult] = []
        if not res or not res.get("ids") or not res["ids"] or not res["ids"][0]:
            return results

        ids = res["ids"][0]
        docs = res["documents"][0] if res.get("documents") and res["documents"] else []
        metas = res["metadatas"][0] if res.get("metadatas") and res["metadatas"] else []
        dists = res["distances"][0] if res.get("distances") and res["distances"] else []

        for i in range(len(ids)):
            meta = metas[i] if metas and i < len(metas) and metas[i] is not None else {}
            dist_val = float(dists[i]) if dists and i < len(dists) and dists[i] is not None else 0.0

            results.append(
                VectorSearchResult(
                    id=ids[i],
                    document_id=meta.get("document_id", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=docs[i] if docs and i < len(docs) else "",
                    filename=meta.get("filename", ""),
                    file_type=meta.get("file_type", ""),
                    page_number=meta.get("page_number"),
                    distance=dist_val,
                )
            )

        return results

    def query_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """Alias for query_nearest."""
        return self.query_nearest(query_vector=query_vector, top_k=top_k, where=where)

    def get_records_by_document(
        self,
        document_id: str,
        include_embeddings: bool = False,
    ) -> list[StoredChunkRecord]:
        """Retrieve all stored chunk records for a given document UUID.

        Args:
            document_id: UUID string of the document.
            include_embeddings: Whether to return dense vectors with the records.

        Returns:
            List of StoredChunkRecord objects sorted by chunk_index ascending.

        Raises:
            ChromaValidationError: If document_id is empty or invalid.
            ChromaStorageError: If retrieval fails.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise ChromaValidationError("document_id must be a non-empty string")

        col = self._get_collection()
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes.append("embeddings")

        try:
            results = col.get(where={"document_id": document_id.strip()}, include=includes)
        except Exception as exc:
            logger.error("Failed to get chunks for document %s: %s", document_id, type(exc).__name__)
            raise ChromaStorageError(
                f"Failed to retrieve chunks for document '{document_id}': {type(exc).__name__}"
            ) from exc

        records: list[StoredChunkRecord] = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        embs = results.get("embeddings")

        for i in range(len(ids)):
            meta = metas[i] if metas and i < len(metas) and metas[i] is not None else {}
            emb_val = None
            if embs is not None and i < len(embs) and embs[i] is not None:
                emb_val = [float(x) for x in embs[i]]

            records.append(
                StoredChunkRecord(
                    id=ids[i],
                    document_id=meta.get("document_id", document_id),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=docs[i] if docs and i < len(docs) else "",
                    filename=meta.get("filename", ""),
                    file_type=meta.get("file_type", ""),
                    page_number=meta.get("page_number"),
                    embedding=emb_val,
                )
            )

        # Sort by chunk_index ascending for deterministic ordering
        records.sort(key=lambda r: r.chunk_index)
        return records

    def get_chunks_by_document_id(
        self,
        document_id: str,
        include_embeddings: bool = False,
    ) -> list[StoredChunkRecord]:
        """Alias for get_records_by_document."""
        return self.get_records_by_document(
            document_id=document_id,
            include_embeddings=include_embeddings,
        )

    def delete_records_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document UUID from ChromaDB.

        Args:
            document_id: UUID string of the document.

        Returns:
            Number of deleted chunks.

        Raises:
            ChromaValidationError: If document_id is invalid.
            ChromaStorageError: If retrieval or deletion fails.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise ChromaValidationError("document_id must be a non-empty string")

        doc_id_clean = document_id.strip()
        col = self._get_collection()
        try:
            # Retrieve the IDs matching the document_id filter before deletion
            existing = col.get(where={"document_id": doc_id_clean}, include=[])
            matching_ids = (
                existing.get("ids", [])
                if existing and isinstance(existing, dict)
                else []
            )
            count = len(matching_ids)

            # Execute the deletion using the existing document_id filter
            col.delete(where={"document_id": doc_id_clean})

            # Return the previously determined count
            return count
        except Exception as exc:
            logger.error("Failed to delete chunks for document %s: %s", document_id, type(exc).__name__)
            raise ChromaStorageError(
                f"Failed to delete chunks for document '{document_id}': {type(exc).__name__}"
            ) from exc

    def delete_chunks_by_document_id(self, document_id: str) -> int:
        """Alias for delete_records_by_document."""
        return self.delete_records_by_document(document_id=document_id)

    def check_readiness(self) -> str:
        """Verify storage readiness for this adapter."""
        return check_chroma_readiness(
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
        )

    def check_health(self) -> bool:
        """Boolean health check confirming collection accessibility."""
        return self.check_readiness() == "ok"

    def count(self) -> int:
        """Return total chunk count in the collection."""
        col = self._get_collection()
        try:
            return col.count()
        except Exception as exc:
            logger.error("Failed to count ChromaDB chunks: %s", type(exc).__name__)
            raise ChromaStorageError(f"Failed to count ChromaDB chunks: {type(exc).__name__}") from exc
