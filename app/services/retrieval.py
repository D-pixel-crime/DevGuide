"""Semantic retrieval service with cosine distance filtering and canonical matching."""

from dataclasses import dataclass
from typing import Any, Optional, Union

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.chroma import ChromaAdapter, VectorSearchResult
from app.services.embedding import OpenAIEmbeddingAdapter

logger = get_logger("app.services.retrieval")


class RetrievalError(Exception):
    """Base exception for retrieval service errors."""


class InvalidQueryError(RetrievalError, ValueError):
    """Raised when query question or top_k parameter fails validation."""


@dataclass(frozen=True)
class RetrievedMatch:
    """Individual retrieved chunk match with canonical metadata and raw distance.

    Attributes:
        text: Raw passage text of the chunk.
        document_id: UUID of parent document.
        filename: Original safe filename.
        file_type: File extension ('pdf', 'md', 'txt').
        chunk_index: 0-based sequential chunk index within document.
        page_number: 1-based page number for PDF, None for Markdown/TXT.
        distance: Raw geometric cosine distance returned by ChromaDB (0.0 to 2.0).
            Smaller cosine distance indicates closer vectors in embedding space.
            It is not a calibrated confidence or probability score.
    """

    text: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    page_number: Optional[int]
    distance: float


@dataclass(frozen=True)
class RetrievalResult:
    """Aggregated retrieval result before any generation or citation tagging.

    Attributes:
        question: The user query string.
        requested_top_k: Resolved number of nearest neighbors requested.
        has_relevant_context: Boolean flag indicating if any chunks met distance threshold.
        retrieved_count: Number of chunks passing the distance threshold.
        matches: List of RetrievedMatch items ordered by cosine distance ascending.
    """

    question: str
    requested_top_k: int
    has_relevant_context: bool
    retrieved_count: int
    matches: list[RetrievedMatch]


class RetrievalService:
    """Semantic vector search service with relevance filtering and metadata mapping.

    Executes query embedding via the configured embedding model, performs nearest-neighbor
    search in ChromaDB, applies MAX_COSINE_DISTANCE as an initial relevance heuristic,
    and maps results to typed matches without assigning generation source labels.
    """

    def __init__(
        self,
        chroma_adapter: Union[ChromaAdapter, Any],
        embedding_adapter: Union[OpenAIEmbeddingAdapter, Any],
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize retrieval service.

        Args:
            chroma_adapter: ChromaDB adapter for vector queries.
            embedding_adapter: Embedding adapter for generating query vectors.
            settings: Optional application settings (defaults to get_settings()).
        """
        self._chroma_adapter = chroma_adapter
        self._embedding_adapter = embedding_adapter
        self._settings = settings if settings is not None else get_settings()

    @property
    def chroma_adapter(self) -> Any:
        """Configured ChromaDB vector store adapter."""
        return self._chroma_adapter

    @property
    def embedding_adapter(self) -> Any:
        """Configured embedding adapter."""
        return self._embedding_adapter

    @property
    def settings(self) -> Settings:
        """Active application settings."""
        return self._settings

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve relevant document chunks for a question using cosine distance.

        Args:
            question: Non-empty query string (max 1000 characters).
            top_k: Optional number of chunks to retrieve (1..MAX_TOP_K). Defaults to DEFAULT_TOP_K.

        Returns:
            RetrievalResult containing matches passing the MAX_COSINE_DISTANCE heuristic.

        Raises:
            TypeError: If question is not a string.
            InvalidQueryError: If question is empty or top_k is out of range [1, MAX_TOP_K].
            EmbeddingError: If embedding generation fails.
            ChromaError: If ChromaDB query fails.
        """
        # 1. Validate question
        if not isinstance(question, str):
            raise TypeError(f"question must be a string, got {type(question).__name__}")

        if len(question) > 1000:
            raise InvalidQueryError("Question cannot exceed 1000 characters")

        clean_question = question.strip()
        if not clean_question:
            raise InvalidQueryError("Question cannot be empty")

        # 2. Validate and resolve top_k
        max_top_k = self._settings.MAX_TOP_K
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > max_top_k:
                raise InvalidQueryError(f"Parameter 'top_k' must be between 1 and {max_top_k}")
            effective_top_k = top_k
        else:
            effective_top_k = self._settings.DEFAULT_TOP_K

        logger.info(
            "Executing semantic retrieval for question (length=%d, top_k=%d)",
            len(clean_question),
            effective_top_k,
        )

        # 3. Embed query using the configured embedding adapter
        query_vector = self._embedding_adapter.embed_query(clean_question)

        # 4. Query ChromaDB for nearest chunks
        raw_results: list[VectorSearchResult] = self._chroma_adapter.query_nearest(
            query_vector=query_vector,
            top_k=effective_top_k,
        )

        # 5. Apply MAX_COSINE_DISTANCE as an initial relevance-filtering heuristic.
        # Smaller cosine distance means closer vectors under the configured metric;
        # it is not a calibrated confidence or probability score.
        # Chunks with distance > MAX_COSINE_DISTANCE are discarded.
        # Raw distances are preserved without claiming empirical validation.
        max_distance = self._settings.MAX_COSINE_DISTANCE
        relevant_results = [r for r in raw_results if r.distance <= max_distance]

        # Sort strictly by distance ascending (closest vectors first)
        relevant_results.sort(key=lambda r: r.distance)

        # 6. Map to canonical RetrievedMatch domain records
        matches = [
            RetrievedMatch(
                text=r.text,
                document_id=r.document_id,
                filename=r.filename,
                file_type=r.file_type,
                chunk_index=r.chunk_index,
                page_number=r.page_number,
                distance=r.distance,
            )
            for r in relevant_results
        ]

        has_relevant_context = len(matches) > 0
        retrieved_count = len(matches)

        logger.info(
            "Retrieval complete: %d raw chunks retrieved, %d relevant chunks passed threshold (<= %.2f)",
            len(raw_results),
            retrieved_count,
            max_distance,
        )

        return RetrievalResult(
            question=clean_question,
            requested_top_k=effective_top_k,
            has_relevant_context=has_relevant_context,
            retrieved_count=retrieved_count,
            matches=matches,
        )
