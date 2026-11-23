"""Comprehensive unit tests for RetrievalService."""

from unittest.mock import MagicMock
import pytest

from app.core.config import Settings
from app.db.chroma import ChromaStorageError, VectorSearchResult
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingProviderError,
    MissingApiKeyError,
    OpenAIEmbeddingAdapter,
)
from app.services.retrieval import (
    InvalidQueryError,
    RetrievalResult,
    RetrievalService,
    RetrievedMatch,
)


class FakeEmbeddingAdapter:
    """Deterministic fake embedding adapter for retrieval unit tests."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 4) -> None:
        self.model = model
        self.dim = dim
        self.embed_query_calls: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        return [0.1] * self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


class FakeChromaAdapter:
    """Deterministic fake ChromaDB adapter for retrieval unit tests."""

    def __init__(self, search_results: list[VectorSearchResult] | None = None) -> None:
        self.search_results = search_results if search_results is not None else []
        self.query_nearest_calls: list[dict] = []

    def query_nearest(
        self, query_vector: list[float], top_k: int = 5, where: dict | None = None
    ) -> list[VectorSearchResult]:
        self.query_nearest_calls.append(
            {"query_vector": query_vector, "top_k": top_k, "where": where}
        )
        return self.search_results[:top_k]


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
    )


# ==============================================================================
# 1. Input Validation Tests (Question & Top_K)
# ==============================================================================


def test_question_empty_or_whitespace_raises(test_settings: Settings) -> None:
    """Validate that empty or whitespace-only questions raise InvalidQueryError."""
    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    with pytest.raises(InvalidQueryError, match="Question cannot be empty"):
        service.retrieve("")

    with pytest.raises(InvalidQueryError, match="Question cannot be empty"):
        service.retrieve("   \t\n  ")


def test_question_non_string_raises(test_settings: Settings) -> None:
    """Validate that non-string questions raise TypeError."""
    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    with pytest.raises(TypeError, match="question must be a string"):
        service.retrieve(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="question must be a string"):
        service.retrieve(12345)  # type: ignore[arg-type]


def test_question_too_long_raises(test_settings: Settings) -> None:
    """Validate that questions exceeding 1000 characters raise InvalidQueryError."""
    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    with pytest.raises(InvalidQueryError, match="Question cannot exceed 1000 characters"):
        service.retrieve("a" * 1001)


def test_question_too_long_before_trimming_raises(test_settings: Settings) -> None:
    """Validate that questions exceeding 1000 characters before trimming raise InvalidQueryError."""
    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    padded = (" " * 200) + "Short content" + (" " * 800)
    assert len(padded) == 1013
    assert len(padded.strip()) == 13

    with pytest.raises(InvalidQueryError, match="Question cannot exceed 1000 characters"):
        service.retrieve(padded)


def test_top_k_bounds_validation(test_settings: Settings) -> None:
    """Validate top_k lower bound, upper bound, non-int, and boolean rejection."""
    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    # Below 1
    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        service.retrieve("Valid question", top_k=0)

    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        service.retrieve("Valid question", top_k=-5)

    # Above MAX_TOP_K (10)
    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        service.retrieve("Valid question", top_k=11)

    # Boolean values
    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        service.retrieve("Valid question", top_k=True)  # type: ignore[arg-type]

    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        service.retrieve("Valid question", top_k=False)  # type: ignore[arg-type]

    # Non-integer values
    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        service.retrieve("Valid question", top_k="5")  # type: ignore[arg-type]


def test_default_top_k_resolution(test_settings: Settings) -> None:
    """Validate that omitting top_k defaults to DEFAULT_TOP_K (5)."""
    fake_chroma = FakeChromaAdapter()
    service = RetrievalService(
        chroma_adapter=fake_chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("How do I start?")
    assert result.requested_top_k == 5
    assert len(fake_chroma.query_nearest_calls) == 1
    assert fake_chroma.query_nearest_calls[0]["top_k"] == 5


def test_custom_top_k_resolution(test_settings: Settings) -> None:
    """Validate that explicit top_k is respected and forwarded to Chroma query."""
    fake_chroma = FakeChromaAdapter()
    service = RetrievalService(
        chroma_adapter=fake_chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("How do I start?", top_k=8)
    assert result.requested_top_k == 8
    assert fake_chroma.query_nearest_calls[0]["top_k"] == 8


# ==============================================================================
# 2. Embedding Model Consistency Tests
# ==============================================================================


def test_embedding_model_consistency(test_settings: Settings) -> None:
    """Verify retrieval uses the configured OPENAI_EMBEDDING_MODEL identically to ingestion."""
    embed_adapter = OpenAIEmbeddingAdapter(settings=test_settings)
    assert embed_adapter.model == test_settings.OPENAI_EMBEDDING_MODEL

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=embed_adapter,
        settings=test_settings,
    )
    assert service.embedding_adapter.model == "text-embedding-3-small"

    # Custom model test
    custom_settings = Settings(OPENAI_EMBEDDING_MODEL="custom-model-v2")
    custom_embed = OpenAIEmbeddingAdapter(settings=custom_settings)
    custom_service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=custom_embed,
        settings=custom_settings,
    )
    assert custom_service.embedding_adapter.model == "custom-model-v2"


# ==============================================================================
# 3. Chroma Result Mapping & Metadata Preservation Tests
# ==============================================================================


def test_chroma_result_mapping_canonical_fields(test_settings: Settings) -> None:
    """Verify Chroma search results are mapped to canonical RetrievedMatch records."""
    mock_results = [
        VectorSearchResult(
            id="doc1:0",
            document_id="doc1",
            chunk_index=0,
            text="First chunk text",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.25,
        ),
        VectorSearchResult(
            id="doc2:3",
            document_id="doc2",
            chunk_index=3,
            text="Second chunk text from PDF",
            filename="guide.pdf",
            file_type="pdf",
            page_number=4,
            distance=0.45,
        ),
    ]

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=mock_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Query about configuration")
    assert result.has_relevant_context is True
    assert result.retrieved_count == 2
    assert len(result.matches) == 2

    # Check match 0 (non-PDF)
    m0 = result.matches[0]
    assert m0.text == "First chunk text"
    assert m0.document_id == "doc1"
    assert m0.filename == "quickstart.md"
    assert m0.file_type == "md"
    assert m0.chunk_index == 0
    assert m0.page_number is None
    assert m0.distance == 0.25

    # Check match 1 (PDF)
    m1 = result.matches[1]
    assert m1.text == "Second chunk text from PDF"
    assert m1.document_id == "doc2"
    assert m1.filename == "guide.pdf"
    assert m1.file_type == "pdf"
    assert m1.chunk_index == 3
    assert m1.page_number == 4
    assert m1.distance == 0.45

    # Confirm no source_label attribute exists
    assert not hasattr(m0, "source_label")
    assert not hasattr(m1, "source_label")


# ==============================================================================
# 4. Relevance Heuristic & Distance Filtering Tests
# ==============================================================================


def test_relevance_threshold_filtering(test_settings: Settings) -> None:
    """Verify chunks with distance > 0.70 are excluded, <= 0.70 are included."""
    mock_results = [
        VectorSearchResult(
            id="d:1",
            document_id="d",
            chunk_index=1,
            text="Close match",
            filename="doc.txt",
            file_type="txt",
            page_number=None,
            distance=0.15,
        ),
        VectorSearchResult(
            id="d:2",
            document_id="d",
            chunk_index=2,
            text="Exact threshold boundary match",
            filename="doc.txt",
            file_type="txt",
            page_number=None,
            distance=0.70,  # Exactly equal to threshold
        ),
        VectorSearchResult(
            id="d:3",
            document_id="d",
            chunk_index=3,
            text="Just over threshold",
            filename="doc.txt",
            file_type="txt",
            page_number=None,
            distance=0.7001,  # Should be excluded
        ),
        VectorSearchResult(
            id="d:4",
            document_id="d",
            chunk_index=4,
            text="Distant chunk",
            filename="doc.txt",
            file_type="txt",
            page_number=None,
            distance=1.20,  # Should be excluded
        ),
    ]

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=mock_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Query test")
    assert result.has_relevant_context is True
    assert result.retrieved_count == 2
    assert [m.distance for m in result.matches] == [0.15, 0.70]


def test_relevance_ordering_distance_ascending(test_settings: Settings) -> None:
    """Verify retrieved matches are strictly sorted ascending by cosine distance."""
    mock_results = [
        VectorSearchResult(
            id="d:2",
            document_id="d",
            chunk_index=2,
            text="Second closest",
            filename="f.md",
            file_type="md",
            page_number=None,
            distance=0.45,
        ),
        VectorSearchResult(
            id="d:1",
            document_id="d",
            chunk_index=1,
            text="Closest",
            filename="f.md",
            file_type="md",
            page_number=None,
            distance=0.12,
        ),
    ]

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=mock_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Query test")
    assert [m.distance for m in result.matches] == [0.12, 0.45]
    assert result.matches[0].text == "Closest"
    assert result.matches[1].text == "Second closest"


def test_raw_distance_preserved_without_transformation(test_settings: Settings) -> None:
    """Verify raw geometric cosine distance is preserved exactly as float."""
    raw_dist = 0.283749182374
    mock_results = [
        VectorSearchResult(
            id="d:1",
            document_id="d",
            chunk_index=0,
            text="Precision distance test",
            filename="f.txt",
            file_type="txt",
            page_number=None,
            distance=raw_dist,
        )
    ]

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=mock_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Precision test")
    assert result.matches[0].distance == raw_dist


# ==============================================================================
# 5. Empty & Zero Relevant Matches Tests
# ==============================================================================


def test_empty_collection_returns_empty_results(test_settings: Settings) -> None:
    """Verify querying an empty collection returns has_relevant_context=False and empty matches."""
    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=[]),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Any question")
    assert result.has_relevant_context is False
    assert result.retrieved_count == 0
    assert result.matches == []
    assert result.requested_top_k == 5


def test_zero_relevant_matches_when_all_exceed_threshold(test_settings: Settings) -> None:
    """Verify querying returns has_relevant_context=False when all chunks exceed 0.70."""
    mock_results = [
        VectorSearchResult(
            id="d:1",
            document_id="d",
            chunk_index=0,
            text="Irrelevant text 1",
            filename="f.md",
            file_type="md",
            page_number=None,
            distance=0.85,
        ),
        VectorSearchResult(
            id="d:2",
            document_id="d",
            chunk_index=1,
            text="Irrelevant text 2",
            filename="f.md",
            file_type="md",
            page_number=None,
            distance=0.99,
        ),
    ]

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=mock_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Unrelated topic query")
    assert result.has_relevant_context is False
    assert result.retrieved_count == 0
    assert result.matches == []


# ==============================================================================
# 6. Assertion That Generation Is Never Invoked
# ==============================================================================


def test_generation_is_never_invoked(test_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify semantic retrieval never touches any generation/chat APIs or models."""
    mock_chat = MagicMock()
    # Mock openai.chat or similar if it were to be imported or called
    monkeypatch.setattr("openai.OpenAI", mock_chat)

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=[]),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    result = service.retrieve("Query testing no generation")
    assert result.question == "Query testing no generation"
    assert mock_chat.call_count == 0


# ==============================================================================
# 7. Error Handling Propagation Tests
# ==============================================================================


def test_embedding_provider_error_propagates(test_settings: Settings) -> None:
    """Verify EmbeddingProviderError propagates out of retrieve."""
    mock_embed = MagicMock()
    mock_embed.embed_query.side_effect = EmbeddingProviderError("Rate limit exceeded")

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=mock_embed,
        settings=test_settings,
    )

    with pytest.raises(EmbeddingProviderError, match="Rate limit exceeded"):
        service.retrieve("Test query")


def test_embedding_auth_error_propagates(test_settings: Settings) -> None:
    """Verify EmbeddingAuthenticationError propagates out of retrieve."""
    mock_embed = MagicMock()
    mock_embed.embed_query.side_effect = EmbeddingAuthenticationError("Invalid API key")

    service = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=mock_embed,
        settings=test_settings,
    )

    with pytest.raises(EmbeddingAuthenticationError, match="Invalid API key"):
        service.retrieve("Test query")


def test_chroma_storage_error_propagates(test_settings: Settings) -> None:
    """Verify ChromaStorageError propagates out of retrieve."""
    mock_chroma = MagicMock()
    mock_chroma.query_nearest.side_effect = ChromaStorageError("Disk error")

    service = RetrievalService(
        chroma_adapter=mock_chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=test_settings,
    )

    with pytest.raises(ChromaStorageError, match="Disk error"):
        service.retrieve("Test query")
