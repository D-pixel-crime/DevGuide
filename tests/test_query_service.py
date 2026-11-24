"""Comprehensive unit and contract tests for QueryService RAG orchestration."""

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock
import pytest

from app.core.config import Settings
from app.db.chroma import ChromaStorageError, VectorSearchResult
from app.models.schemas import QuerySource
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingProviderError,
    MissingApiKeyError,
)
from app.services.generation import (
    CANONICAL_ABSTENTION_ANSWER,
    AnswerContextBuilder,
    BuiltContext,
    GenerationAuthenticationError,
    GenerationError,
    GenerationMissingApiKeyError,
    GenerationProviderError,
    GenerationResult,
    InvalidGenerationResponseError,
    OpenAIGenerationAdapter,
)
from app.services.query import QueryResult, QueryService
from app.services.retrieval import (
    InvalidQueryError,
    RetrievalResult,
    RetrievalService,
    RetrievedMatch,
)


# ==============================================================================
# Test Doubles & Fakes
# ==============================================================================


class FakeEmbeddingAdapter:
    """Deterministic fake embedding adapter for query unit tests."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 4) -> None:
        self.model = model
        self.dim = dim
        self.embed_query_calls: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        return [0.1] * self.dim


class FakeChromaAdapter:
    """Deterministic fake ChromaDB adapter for query unit tests."""

    def __init__(self, search_results: Optional[list[VectorSearchResult]] = None) -> None:
        self.search_results = search_results if search_results is not None else []
        self.query_nearest_calls: list[dict] = []

    def query_nearest(
        self, query_vector: list[float], top_k: int = 5, where: Optional[dict] = None
    ) -> list[VectorSearchResult]:
        self.query_nearest_calls.append(
            {"query_vector": query_vector, "top_k": top_k, "where": where}
        )
        return self.search_results[:top_k]


class FakeGenerationAdapter:
    """Deterministic fake generation adapter tracking invocation count and arguments."""

    def __init__(
        self,
        canned_answer: str = "Start the server using `uvicorn app.main:app`. [S1]",
        model: str = "gpt-6-luna",
        exception_to_raise: Optional[Exception] = None,
    ) -> None:
        self.canned_answer = canned_answer
        self.model = model
        self.exception_to_raise = exception_to_raise
        self.generate_calls: list[dict[str, Any]] = []

    def generate(
        self,
        question: str,
        context: Any,
        system_instructions: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        self.generate_calls.append(
            {
                "question": question,
                "context": context,
                "system_instructions": system_instructions,
                "model": model,
            }
        )
        if self.exception_to_raise is not None:
            raise self.exception_to_raise
        return GenerationResult(
            answer=self.canned_answer,
            model=model or self.model,
            raw_response_id="resp_fake_123",
        )


@pytest.fixture
def default_settings() -> Settings:
    return Settings(
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
        MAX_CONTEXT_CHARS=12000,
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
        OPENAI_CHAT_MODEL="gpt-6-luna",
    )


# ==============================================================================
# 1. Grounded Generation Tests
# ==============================================================================


def test_query_successful_grounded_answer(default_settings: Settings) -> None:
    """Verify successful RAG query returns grounded answer with structured QuerySource objects."""
    chroma_results = [
        VectorSearchResult(
            id="doc1:0",
            document_id="doc1-uuid",
            chunk_index=0,
            text="To start the API server, run `taskboard api start --port 8080`.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.22,
        ),
        VectorSearchResult(
            id="doc2:2",
            document_id="doc2-uuid",
            chunk_index=2,
            text="Physical PDF manual instructions on server configuration.",
            filename="manual.pdf",
            file_type="pdf",
            page_number=5,
            distance=0.45,
        ),
    ]

    chroma = FakeChromaAdapter(search_results=chroma_results)
    embedder = FakeEmbeddingAdapter()
    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=embedder,
        settings=default_settings,
    )

    fake_gen = FakeGenerationAdapter(
        canned_answer="Run the documented startup command. [S1]"
    )

    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=fake_gen,
        settings=default_settings,
    )

    result = query_svc.query("How do I start the API server?", top_k=5)

    # 1. Check QueryResult properties
    assert isinstance(result, QueryResult)
    assert result.question == "How do I start the API server?"
    assert result.answer == "Run the documented startup command. [S1]"
    assert result.abstained is False
    assert result.retrieved_count == 2
    assert len(result.sources) == 2

    # 2. Check source 0 (Markdown, page_number is None)
    s0 = result.sources[0]
    assert isinstance(s0, QuerySource)
    assert s0.source_label == "S1"
    assert s0.document_id == "doc1-uuid"
    assert s0.filename == "quickstart.md"
    assert s0.file_type == "md"
    assert s0.page_number is None
    assert s0.chunk_index == 0
    assert s0.distance == 0.22
    assert s0.excerpt == "To start the API server, run `taskboard api start --port 8080`."

    # 3. Check source 1 (PDF, page_number is 5)
    s1 = result.sources[1]
    assert isinstance(s1, QuerySource)
    assert s1.source_label == "S2"
    assert s1.document_id == "doc2-uuid"
    assert s1.filename == "manual.pdf"
    assert s1.file_type == "pdf"
    assert s1.page_number == 5
    assert s1.chunk_index == 2
    assert s1.distance == 0.45
    assert s1.excerpt == "Physical PDF manual instructions on server configuration."

    # 4. Verify generation adapter was called exactly once
    assert len(fake_gen.generate_calls) == 1
    call = fake_gen.generate_calls[0]
    assert call["question"] == "How do I start the API server?"
    assert isinstance(call["context"], BuiltContext)
    assert call["context"].included_count == 2


# ==============================================================================
# 2. Abstention Tests & Generation Skipped Verification
# ==============================================================================


def test_query_abstention_when_no_chunks_pass_threshold(default_settings: Settings) -> None:
    """Verify pipeline abstains and skips generation when all chunks exceed MAX_COSINE_DISTANCE (0.70)."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc-distant",
            chunk_index=0,
            text="Irrelevant text about quantum physics.",
            filename="notes.txt",
            file_type="txt",
            page_number=None,
            distance=0.88,  # > 0.70
        ),
        VectorSearchResult(
            id="d:2",
            document_id="doc-distant-2",
            chunk_index=1,
            text="Another irrelevant chunk.",
            filename="notes2.txt",
            file_type="txt",
            page_number=None,
            distance=1.15,  # > 0.70
        ),
    ]

    chroma = FakeChromaAdapter(search_results=chroma_results)
    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )

    fake_gen = FakeGenerationAdapter()
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=fake_gen,
        settings=default_settings,
    )

    result = query_svc.query("What is the automated database backup schedule?")

    # 1. Exact canonical abstention contract
    assert result.question == "What is the automated database backup schedule?"
    assert result.answer == CANONICAL_ABSTENTION_ANSWER
    assert result.answer == "I do not have enough relevant information in the documentation to answer your question."
    assert result.abstained is True
    assert result.retrieved_count == 0
    assert result.sources == []

    # 2. CRITICAL: Generation model must NOT be called on abstention
    assert len(fake_gen.generate_calls) == 0


def test_query_abstention_on_empty_collection(default_settings: Settings) -> None:
    """Verify pipeline abstains without calling generation when collection is completely empty."""
    chroma = FakeChromaAdapter(search_results=[])
    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )

    fake_gen = FakeGenerationAdapter()
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=fake_gen,
        settings=default_settings,
    )

    result = query_svc.query("Any valid question?")

    assert result.abstained is True
    assert result.answer == CANONICAL_ABSTENTION_ANSWER
    assert result.retrieved_count == 0
    assert result.sources == []
    assert len(fake_gen.generate_calls) == 0


def test_query_abstention_when_context_budget_omits_all_chunks(default_settings: Settings) -> None:
    """Verify abstention occurs if context budget ceiling omits all candidates."""
    tiny_settings = Settings(
        MAX_CONTEXT_CHARS=10,  # Extremely tiny limit
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
    )
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Long text that cannot possibly fit in 10 characters.",
            filename="doc.txt",
            file_type="txt",
            page_number=None,
            distance=0.20,
        ),
    ]

    chroma = FakeChromaAdapter(search_results=chroma_results)
    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=tiny_settings,
    )

    fake_gen = FakeGenerationAdapter()
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=fake_gen,
        settings=tiny_settings,
    )

    result = query_svc.query("Valid query?")
    assert result.abstained is True
    assert result.answer == CANONICAL_ABSTENTION_ANSWER
    assert result.retrieved_count == 0
    assert result.sources == []
    assert len(fake_gen.generate_calls) == 0


# ==============================================================================
# 3. Context Bounds & Source Metadata Integrity
# ==============================================================================


def test_context_bounds_and_omitted_chunks(default_settings: Settings) -> None:
    """Verify source labels map only to chunks that fit within MAX_CONTEXT_CHARS."""
    # Chunk text is 200 chars each. With header ~60 chars, each block is ~260 chars.
    chunk1_text = "A" * 200
    chunk2_text = "B" * 200
    chunk3_text = "C" * 200

    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text=chunk1_text,
            filename="f1.md",
            file_type="md",
            page_number=None,
            distance=0.10,
        ),
        VectorSearchResult(
            id="d:2",
            document_id="doc2",
            chunk_index=1,
            text=chunk2_text,
            filename="f2.md",
            file_type="md",
            page_number=None,
            distance=0.20,
        ),
        VectorSearchResult(
            id="d:3",
            document_id="doc3",
            chunk_index=2,
            text=chunk3_text,
            filename="f3.md",
            file_type="md",
            page_number=None,
            distance=0.30,
        ),
    ]

    # Set context limit to only allow 2 blocks (~600 chars)
    budget_settings = Settings(
        MAX_CONTEXT_CHARS=600,
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
    )

    chroma = FakeChromaAdapter(search_results=chroma_results)
    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=budget_settings,
    )

    fake_gen = FakeGenerationAdapter()
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=fake_gen,
        settings=budget_settings,
    )

    result = query_svc.query("Question")
    assert result.abstained is False
    assert result.retrieved_count == 2
    assert len(result.sources) == 2
    assert result.sources[0].source_label == "S1"
    assert result.sources[0].document_id == "doc1"
    assert result.sources[1].source_label == "S2"
    assert result.sources[1].document_id == "doc2"

    # doc3 was omitted and must NOT appear in sources
    doc_ids = [s.document_id for s in result.sources]
    assert "doc3" not in doc_ids


# ==============================================================================
# 4. Input Validation & Parameter Forwarding
# ==============================================================================


def test_query_validation_empty_question(default_settings: Settings) -> None:
    """Verify empty or whitespace-only questions raise InvalidQueryError."""
    query_svc = QueryService(
        retrieval_service=RetrievalService(FakeChromaAdapter(), FakeEmbeddingAdapter(), default_settings),
        generation_adapter=FakeGenerationAdapter(),
        settings=default_settings,
    )

    with pytest.raises(InvalidQueryError, match="Question cannot be empty"):
        query_svc.query("")

    with pytest.raises(InvalidQueryError, match="Question cannot be empty"):
        query_svc.query("   \t  ")


def test_query_validation_question_length(default_settings: Settings) -> None:
    """Verify questions exceeding 1000 characters raise InvalidQueryError."""
    query_svc = QueryService(
        retrieval_service=RetrievalService(FakeChromaAdapter(), FakeEmbeddingAdapter(), default_settings),
        generation_adapter=FakeGenerationAdapter(),
        settings=default_settings,
    )

    with pytest.raises(InvalidQueryError, match="Question cannot exceed 1000 characters"):
        query_svc.query("x" * 1001)


def test_query_validation_top_k_bounds(default_settings: Settings) -> None:
    """Verify top_k < 1 or > 10 raises InvalidQueryError."""
    query_svc = QueryService(
        retrieval_service=RetrievalService(FakeChromaAdapter(), FakeEmbeddingAdapter(), default_settings),
        generation_adapter=FakeGenerationAdapter(),
        settings=default_settings,
    )

    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        query_svc.query("Valid question", top_k=0)

    with pytest.raises(InvalidQueryError, match="Parameter 'top_k' must be between 1 and 10"):
        query_svc.query("Valid question", top_k=11)


def test_query_top_k_forwarded_to_retrieval(default_settings: Settings) -> None:
    """Verify top_k is forwarded correctly to underlying retrieval service."""
    chroma = FakeChromaAdapter()
    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=FakeGenerationAdapter(),
        settings=default_settings,
    )

    query_svc.query("Valid question", top_k=7)
    assert len(chroma.query_nearest_calls) == 1
    assert chroma.query_nearest_calls[0]["top_k"] == 7


# ==============================================================================
# 5. Error Propagation Tests
# ==============================================================================


def test_query_embedding_provider_error_propagates(default_settings: Settings) -> None:
    """Verify upstream embedding provider errors propagate out of query."""
    mock_embed = MagicMock()
    mock_embed.embed_query.side_effect = EmbeddingProviderError("OpenAI API connection timeout")

    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(),
        embedding_adapter=mock_embed,
        settings=default_settings,
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=FakeGenerationAdapter(),
        settings=default_settings,
    )

    with pytest.raises(EmbeddingProviderError, match="connection timeout"):
        query_svc.query("Valid question")


def test_query_generation_provider_error_propagates(default_settings: Settings) -> None:
    """Verify upstream generation provider errors propagate out of query."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Relevant text",
            filename="f.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )

    failing_gen = FakeGenerationAdapter(
        exception_to_raise=GenerationProviderError("OpenAI rate limit exceeded during answer generation")
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=failing_gen,
        settings=default_settings,
    )

    with pytest.raises(GenerationProviderError, match="rate limit exceeded"):
        query_svc.query("Valid question")


def test_query_chroma_storage_error_propagates(default_settings: Settings) -> None:
    """Verify ChromaDB storage errors propagate out of query."""
    mock_chroma = MagicMock()
    mock_chroma.query_nearest.side_effect = ChromaStorageError("Disk read failure")

    retrieval_svc = RetrievalService(
        chroma_adapter=mock_chroma,
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=FakeGenerationAdapter(),
        settings=default_settings,
    )

    with pytest.raises(ChromaStorageError, match="Disk read failure"):
        query_svc.query("Valid question")


def test_query_missing_api_key_propagates(default_settings: Settings) -> None:
    """Verify MissingApiKeyError during generation propagates."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Relevant text",
            filename="f.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )

    failing_gen = FakeGenerationAdapter(
        exception_to_raise=GenerationMissingApiKeyError("OPENAI_API_KEY is not set or empty")
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=failing_gen,
        settings=default_settings,
    )

    with pytest.raises(GenerationMissingApiKeyError, match="OPENAI_API_KEY is not set"):
        query_svc.query("Valid question")


# ==============================================================================
# 6. Generated Citation Validation Tests (Correction 1)
# ==============================================================================


def test_citation_validation_valid_single_citation(default_settings: Settings) -> None:
    """Verify that an answer citing valid [S1] succeeds and preserves the answer text."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(canned_answer="The API server runs on port 8080. [S1]")
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    result = query_svc.query("What is the port?")
    assert result.abstained is False
    assert result.answer == "The API server runs on port 8080. [S1]"
    assert result.retrieved_count == 1
    assert result.sources[0].source_label == "S1"


def test_citation_validation_valid_multiple_citations(default_settings: Settings) -> None:
    """Verify that an answer citing multiple valid labels [S1] and [S2] succeeds."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        ),
        VectorSearchResult(
            id="d:2",
            document_id="doc2",
            chunk_index=1,
            text="Timeout is 30.",
            filename="troubleshooting.md",
            file_type="md",
            page_number=None,
            distance=0.25,
        ),
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="Start on port 8080 [S1], and set database timeout to 30 [S2]."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    result = query_svc.query("How to configure?")
    assert result.abstained is False
    assert result.answer == "Start on port 8080 [S1], and set database timeout to 30 [S2]."
    assert result.retrieved_count == 2
    assert len(result.sources) == 2


def test_citation_validation_unknown_citation_rejected(default_settings: Settings) -> None:
    """Verify that an answer citing an unknown label [S99] raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(canned_answer="The API server runs on port 8080. [S99]")
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"Generated answer cited invalid or omitted source label '\[S99\]'"):
        query_svc.query("What is the port?")


def test_citation_validation_omitted_source_citation_rejected(default_settings: Settings) -> None:
    """Verify that an answer citing a source omitted due to context budget raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="A" * 200,
            filename="f1.md",
            file_type="md",
            page_number=None,
            distance=0.10,
        ),
        VectorSearchResult(
            id="d:2",
            document_id="doc2",
            chunk_index=1,
            text="B" * 200,
            filename="f2.md",
            file_type="md",
            page_number=None,
            distance=0.20,
        ),
        VectorSearchResult(
            id="d:3",
            document_id="doc3",
            chunk_index=2,
            text="C" * 200,
            filename="f3.md",
            file_type="md",
            page_number=None,
            distance=0.30,
        ),
    ]
    budget_settings = Settings(
        MAX_CONTEXT_CHARS=600,  # Fits only 2 chunks (~520 chars)
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
    )
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=budget_settings,
    )
    # The model attempts to cite [S3], which was omitted from the context budget
    generator = FakeGenerationAdapter(canned_answer="According to [S1] and [S3], the setting is configured.")
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=budget_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"Generated answer cited invalid or omitted source label '\[S3\]'"):
        query_svc.query("Question")


def test_citation_validation_normal_answer_no_citation_rejected(default_settings: Settings) -> None:
    """Verify that a normal non-abstention answer with no citations raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(canned_answer="The API server runs on port 8080.")
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match="Generated answer contains no source citations"):
        query_svc.query("What is the port?")


def test_citation_validation_canonical_abstention_allowed_without_citation(default_settings: Settings) -> None:
    """Verify that exact CANONICAL_ABSTENTION_ANSWER is allowed without citations and returns abstained=True."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Some marginally relevant passage.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.65,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(canned_answer=CANONICAL_ABSTENTION_ANSWER)
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    result = query_svc.query("Specific unanswerable detail?")
    assert result.abstained is True
    assert result.answer == CANONICAL_ABSTENTION_ANSWER
    assert result.retrieved_count == 0
    assert result.sources == []


def test_citation_validation_malformed_citation_never_returns_grounded_result(default_settings: Settings) -> None:
    """Verify that any malformed citation output never returns a successful grounded QueryResult."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )

    malformed_answers = [
        "No citations here at all.",
        "Fabricated citation [S0].",
        "Out of range citation [S2].",
        "Invented bracket [S999].",
        "Wrong letter citation [C1].",
        "Malformed spacing citation [S 1].",
        "Lowercase citation [s1].",
    ]

    for bad_answer in malformed_answers:
        generator = FakeGenerationAdapter(canned_answer=bad_answer)
        query_svc = QueryService(
            retrieval_service=retrieval_svc,
            generation_adapter=generator,
            settings=default_settings,
        )

        with pytest.raises(InvalidGenerationResponseError):
            query_svc.query("What is the port?")


def test_citation_validation_valid_s1_combined_with_malformed_c1(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with malformed [C1] raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="The server runs on port 8080 [S1]. Additional evidence: [C1]."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"malformed citation token '\[C1\]'"):
        query_svc.query("What is the port?")


def test_citation_validation_valid_s1_combined_with_malformed_s_space_2(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with malformed spacing [S 2] raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        ),
        VectorSearchResult(
            id="d:2",
            document_id="doc2",
            chunk_index=1,
            text="Timeout is 30.",
            filename="troubleshooting.md",
            file_type="md",
            page_number=None,
            distance=0.25,
        ),
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="The server runs on port 8080 [S1], and timeout is 30 [S 2]."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"malformed citation token '\[S 2\]'"):
        query_svc.query("How to configure?")


def test_citation_validation_valid_s1_combined_with_lowercase_s1(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with lowercase [s1] raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="The server runs on port 8080 [S1], and note [s1]."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"malformed citation token '\[s1\]'"):
        query_svc.query("What is the port?")


def test_citation_validation_valid_s1_combined_with_unknown_s99(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with unknown [S99] raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="The server runs on port 8080 [S1], and extra details [S99]."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"invalid or omitted source label '\[S99\]'"):
        query_svc.query("What is the port?")


def test_citation_validation_ordinary_bracketed_text_allowed(default_settings: Settings) -> None:
    """Verify that ordinary bracketed text like [production] or [optional] is allowed alongside valid citation."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    good_answer = "For [production] environments, the server runs on port 8080 [S1] with [optional] flags."
    generator = FakeGenerationAdapter(canned_answer=good_answer)
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    result = query_svc.query("What is the port?")
    assert result.abstained is False
    assert result.answer == good_answer
    assert result.retrieved_count == 1
    assert result.sources[0].source_label == "S1"


def test_citation_validation_valid_s1_with_unknown_linked_citation(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with unknown linked citation [S99](url) raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="The server runs on port 8080 [S1]. See also [S99](https://example.com)."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"invalid or omitted source label '\[S99\]'"):
        query_svc.query("What is the port?")


def test_citation_validation_valid_s1_with_malformed_linked_citation(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with malformed linked citation [C1](url) raises InvalidGenerationResponseError."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    generator = FakeGenerationAdapter(
        canned_answer="The server runs on port 8080 [S1]. Additional evidence: [C1](https://example.com)."
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    with pytest.raises(InvalidGenerationResponseError, match=r"malformed citation token '\[C1\]'"):
        query_svc.query("What is the port?")


def test_citation_validation_valid_s1_with_ordinary_markdown_link(default_settings: Settings) -> None:
    """Verify that valid [S1] combined with an ordinary Markdown link [documentation](url) succeeds."""
    chroma_results = [
        VectorSearchResult(
            id="d:1",
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            filename="quickstart.md",
            file_type="md",
            page_number=None,
            distance=0.15,
        )
    ]
    retrieval_svc = RetrievalService(
        chroma_adapter=FakeChromaAdapter(search_results=chroma_results),
        embedding_adapter=FakeEmbeddingAdapter(),
        settings=default_settings,
    )
    good_answer = "For details see [documentation](https://example.com), server runs on port 8080 [S1]."
    generator = FakeGenerationAdapter(canned_answer=good_answer)
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=default_settings,
    )

    result = query_svc.query("What is the port?")
    assert result.abstained is False
    assert result.answer == good_answer
    assert result.retrieved_count == 1
    assert result.sources[0].source_label == "S1"



