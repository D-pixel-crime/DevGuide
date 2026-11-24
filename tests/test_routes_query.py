"""Comprehensive tests for end-to-end RAG question answering API endpoint (POST /api/v1/query)."""

import pathlib
from typing import Any, Optional
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from app.api.routes_query import get_query_service
from app.core.config import Settings
from app.db.chroma import ChromaAdapter, ChromaStorageError, ChunkRecord, VectorSearchResult
from app.db.sqlite import init_db
from app.main import create_app
from app.models.schemas import QueryRequest, QueryResponse, QuerySource
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    MissingApiKeyError,
)
from app.services.generation import (
    CANONICAL_ABSTENTION_ANSWER,
    AnswerContextBuilder,
    GenerationAuthenticationError,
    GenerationError,
    GenerationMissingApiKeyError,
    GenerationProviderError,
    GenerationResult,
    InvalidGenerationResponseError,
    OpenAIGenerationAdapter,
)
from app.services.query import QueryResult, QueryService
from app.services.retrieval import InvalidQueryError, RetrievalService


# ==============================================================================
# Fakes & Fixtures
# ==============================================================================


class FakeEmbeddingAdapter:
    """Deterministic fake embedding adapter for endpoint tests."""

    def __init__(self, dim: int = 4, model: str = "text-embedding-3-small") -> None:
        self.dim = dim
        self.model = model
        self.queries_embedded: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries_embedded.append(query)
        return [0.1] * self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


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
def query_client(tmp_path: pathlib.Path):
    """Fixture providing TestClient with isolated temporary databases, fake embedding and generation."""
    db_path = tmp_path / "devguide.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path, tmp_path)

    settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_path,
        CHROMA_PATH=chroma_path,
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
        MAX_CONTEXT_CHARS=12000,
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
        OPENAI_CHAT_MODEL="gpt-6-luna",
    )

    chroma = ChromaAdapter(chroma_path=chroma_path, settings=settings)
    chroma.init_collection()
    embedder = FakeEmbeddingAdapter(dim=4)
    generator = FakeGenerationAdapter()

    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=embedder,
        settings=settings,
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=settings,
    )

    app = create_app(settings=settings)
    app.dependency_overrides[get_query_service] = lambda: query_svc

    with TestClient(app) as client:
        yield client, query_svc, chroma, embedder, generator, settings


# ==============================================================================
# 1. Successful Query & Grounded Answer Tests (200 OK)
# ==============================================================================


def test_post_query_200_grounded_answer(query_client) -> None:
    """Verify POST /api/v1/query returns 200 with grounded answer and canonical source objects."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    # Seed ChromaDB with chunks (one markdown, one PDF)
    chunks = [
        ChunkRecord(
            document_id="f8a12c34-90ab-4cde-8123-112233445566",
            chunk_index=0,
            text="To start the API server, run `taskboard api start --port 8080`.",
            embedding=[0.1, 0.1, 0.1, 0.1],  # Exact query vector -> distance ~ 0.0
            filename="quickstart.md",
            file_type="md",
            page_number=None,
        ),
        ChunkRecord(
            document_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            chunk_index=1,
            text="Detailed PDF instructions on port configuration.",
            embedding=[0.1, 0.1, 0.1, 0.12],  # Close vector -> distance < 0.70
            filename="manual.pdf",
            file_type="pdf",
            page_number=4,
        ),
    ]
    chroma.add_chunks(chunks)

    payload = {
        "question": "How do I start the API server?",
        "top_k": 5,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Top-level response structure
    assert data["question"] == "How do I start the API server?"
    assert data["answer"] == "Start the server using `uvicorn app.main:app`. [S1]"
    assert data["abstained"] is False
    assert data["retrieved_count"] == 2
    assert len(data["sources"]) == 2

    # Strict rejection of obsolete/misplaced fields
    assert "query" not in data
    assert "matches" not in data
    assert "results" not in data
    assert "total_retrieved" not in data

    # 2. Source 0 (Markdown: page_number is null)
    s0 = data["sources"][0]
    assert s0["source_label"] == "S1"
    assert s0["document_id"] == "f8a12c34-90ab-4cde-8123-112233445566"
    assert s0["filename"] == "quickstart.md"
    assert s0["file_type"] == "md"
    assert s0["page_number"] is None
    assert s0["chunk_index"] == 0
    assert isinstance(s0["distance"], float)
    assert s0["distance"] <= 0.70
    assert s0["excerpt"] == "To start the API server, run `taskboard api start --port 8080`."

    # 3. Source 1 (PDF: page_number is 4)
    s1 = data["sources"][1]
    assert s1["source_label"] == "S2"
    assert s1["document_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert s1["filename"] == "manual.pdf"
    assert s1["file_type"] == "pdf"
    assert s1["page_number"] == 4
    assert s1["chunk_index"] == 1
    assert isinstance(s1["distance"], float)
    assert s1["distance"] <= 0.70

    # 4. Confirm generator was invoked exactly once
    assert len(generator.generate_calls) == 1


def test_post_query_200_default_top_k(query_client) -> None:
    """Verify that omitting top_k defaults to DEFAULT_TOP_K (5)."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    response = client.post(
        "/api/v1/query",
        json={"question": "What is the timeout?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is the timeout?"


def test_post_query_200_custom_top_k(query_client) -> None:
    """Verify that specifying custom top_k is respected."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    response = client.post(
        "/api/v1/query",
        json={"question": "What is the timeout?", "top_k": 3},
    )
    assert response.status_code == 200


# ==============================================================================
# 2. Abstention Tests (200 OK with abstained=True)
# ==============================================================================


def test_post_query_200_abstention_no_relevant_chunks(query_client) -> None:
    """Verify POST /api/v1/query abstains and skips generation when chunks exceed distance threshold."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    # Add chunk with orthogonal vector (distance > 0.70)
    chunk = ChunkRecord(
        document_id="doc-distant",
        chunk_index=0,
        text="Completely irrelevant content.",
        embedding=[-0.1, -0.1, -0.1, -0.1],
        filename="unrelated.txt",
        file_type="txt",
    )
    chroma.add_chunks([chunk])

    payload = {
        "question": "What is the automated database backup schedule?",
        "top_k": 5,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Exact canonical contract
    assert data["question"] == "What is the automated database backup schedule?"
    assert data["answer"] == CANONICAL_ABSTENTION_ANSWER
    assert data["answer"] == "I do not have enough relevant information in the documentation to answer your question."
    assert data["abstained"] is True
    assert data["retrieved_count"] == 0
    assert data["sources"] == []

    # CRITICAL: generation adapter MUST NOT be called
    assert len(generator.generate_calls) == 0


def test_post_query_200_abstention_empty_collection(query_client) -> None:
    """Verify POST /api/v1/query returns abstention on an empty collection without model call."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    response = client.post(
        "/api/v1/query",
        json={"question": "What is the data retention policy for completed tasks?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["abstained"] is True
    assert data["answer"] == CANONICAL_ABSTENTION_ANSWER
    assert data["retrieved_count"] == 0
    assert data["sources"] == []
    assert len(generator.generate_calls) == 0


# ==============================================================================
# 3. Validation Errors (400 Bad Request & 422 Unprocessable Entity)
# ==============================================================================


def test_post_query_400_empty_question(query_client) -> None:
    """Verify empty question string returns 400 Bad Request."""
    client, _, _, _, _, _ = query_client

    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_post_query_400_whitespace_question(query_client) -> None:
    """Verify whitespace-only question string returns 400 Bad Request."""
    client, _, _, _, _, _ = query_client

    response = client.post("/api/v1/query", json={"question": "   \n\t  "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_post_query_400_question_too_long(query_client) -> None:
    """Verify question exceeding 1000 characters returns 400 Bad Request."""
    client, _, _, _, _, _ = query_client

    response = client.post("/api/v1/query", json={"question": "x" * 1001})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot exceed 1000 characters"


def test_post_query_400_top_k_out_of_bounds(query_client) -> None:
    """Verify top_k < 1 and top_k > 10 return 400 Bad Request."""
    client, _, _, _, _, _ = query_client

    res_low = client.post("/api/v1/query", json={"question": "Valid", "top_k": 0})
    assert res_low.status_code == 400
    assert res_low.json()["detail"] == "Parameter 'top_k' must be between 1 and 10"

    res_high = client.post("/api/v1/query", json={"question": "Valid", "top_k": 11})
    assert res_high.status_code == 400
    assert res_high.json()["detail"] == "Parameter 'top_k' must be between 1 and 10"


def test_post_query_422_missing_question(query_client) -> None:
    """Verify missing required question field returns 422 Unprocessable Entity."""
    client, _, _, _, _, _ = query_client

    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_post_query_422_top_k_invalid_types(query_client) -> None:
    """Verify boolean, string, float top_k values return 422 Unprocessable Entity."""
    client, _, _, _, _, _ = query_client

    assert client.post("/api/v1/query", json={"question": "Valid", "top_k": True}).status_code == 422
    assert client.post("/api/v1/query", json={"question": "Valid", "top_k": False}).status_code == 422
    assert client.post("/api/v1/query", json={"question": "Valid", "top_k": "5"}).status_code == 422
    assert client.post("/api/v1/query", json={"question": "Valid", "top_k": 5.5}).status_code == 422


# ==============================================================================
# 4. Dependency & Provider Errors (503 Service Unavailable)
# ==============================================================================


def test_post_query_503_upstream_embedding_error(query_client) -> None:
    """Verify upstream OpenAI provider errors during embedding return 503."""
    client, _, _, _, _, _ = query_client

    mock_service = MagicMock(spec=QueryService)
    mock_service.query.side_effect = EmbeddingProviderError("OpenAI API connection timeout")

    app = client.app
    app.dependency_overrides[get_query_service] = lambda: mock_service

    response = client.post("/api/v1/query", json={"question": "Query"})
    assert response.status_code == 503
    assert response.json()["detail"] == "OpenAI API connection timeout"


def test_post_query_503_upstream_generation_error(query_client) -> None:
    """Verify upstream OpenAI provider errors during generation return 503."""
    client, _, _, _, _, _ = query_client

    mock_service = MagicMock(spec=QueryService)
    mock_service.query.side_effect = GenerationProviderError("OpenAI rate limit exceeded during answer generation")

    app = client.app
    app.dependency_overrides[get_query_service] = lambda: mock_service

    response = client.post("/api/v1/query", json={"question": "Query"})
    assert response.status_code == 503
    assert response.json()["detail"] == "OpenAI rate limit exceeded during answer generation"


def test_post_query_503_auth_error(query_client) -> None:
    """Verify OpenAI authentication errors return 503 Service Unavailable."""
    client, _, _, _, _, _ = query_client

    mock_service = MagicMock(spec=QueryService)
    mock_service.query.side_effect = GenerationAuthenticationError("OpenAI authentication failed: invalid or unauthorized API key")

    app = client.app
    app.dependency_overrides[get_query_service] = lambda: mock_service

    response = client.post("/api/v1/query", json={"question": "Query"})
    assert response.status_code == 503
    assert response.json()["detail"] == "OpenAI authentication failed: invalid or unauthorized API key"


# ==============================================================================
# 5. Storage, Internal & Malformed Generation Errors (500 Internal Server Error)
# ==============================================================================


def test_post_query_500_chroma_storage_error(query_client) -> None:
    """Verify internal ChromaDB storage failures return 500 without leaking internals."""
    client, _, _, _, _, _ = query_client

    mock_service = MagicMock(spec=QueryService)
    mock_service.query.side_effect = ChromaStorageError("ChromaDB disk read failed")

    app = client.app
    app.dependency_overrides[get_query_service] = lambda: mock_service

    response = client.post("/api/v1/query", json={"question": "Query"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Storage or generation service failure during query"


def test_post_query_500_malformed_generation_response(query_client) -> None:
    """Verify malformed/empty provider response returns 500 without leaking internals."""
    client, _, _, _, _, _ = query_client

    mock_service = MagicMock(spec=QueryService)
    mock_service.query.side_effect = InvalidGenerationResponseError("Response was empty")

    app = client.app
    app.dependency_overrides[get_query_service] = lambda: mock_service

    response = client.post("/api/v1/query", json={"question": "Query"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Generation service returned an invalid or empty response"


def test_post_query_500_missing_api_key(query_client) -> None:
    """Verify MissingApiKeyError returns 500 with safe generic detail."""
    client, _, _, _, _, _ = query_client

    mock_service = MagicMock(spec=QueryService)
    mock_service.query.side_effect = GenerationMissingApiKeyError("OPENAI_API_KEY is not set or empty")

    app = client.app
    app.dependency_overrides[get_query_service] = lambda: mock_service

    response = client.post("/api/v1/query", json={"question": "Query"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Storage or generation service failure during query"


# ==============================================================================
# 6. OpenAPI Specification Verification
# ==============================================================================


def test_openapi_contains_query_endpoint(query_client) -> None:
    """Verify that OpenAPI schema defines POST /api/v1/query correctly."""
    client, _, _, _, _, _ = query_client

    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()

    paths = openapi["paths"]
    assert "/api/v1/query" in paths
    query_op = paths["/api/v1/query"]["post"]

    assert "query" in query_op["tags"]
    assert "200" in query_op["responses"]
    assert "400" in query_op["responses"]
    assert "500" in query_op["responses"]
    assert "503" in query_op["responses"]

    # Request body schema references QueryRequest
    req_schema_ref = query_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert "QueryRequest" in req_schema_ref
    req_model = openapi["components"]["schemas"]["QueryRequest"]

    # question is a required string declaring maxLength=1000
    assert "question" in req_model["required"]
    q_prop = req_model["properties"]["question"]
    assert q_prop["type"] == "string"
    assert q_prop["maxLength"] == 1000

    # top_k is an optional integer declaring minimum=1 and maximum=10
    assert "top_k" not in req_model.get("required", [])
    top_k_prop = req_model["properties"]["top_k"]
    top_k_types = [t.get("type") for t in top_k_prop.get("anyOf", [])]
    assert "integer" in top_k_types or top_k_prop.get("type") == "integer"
    assert top_k_prop["minimum"] in (1, 1.0)
    assert top_k_prop["maximum"] in (10, 10.0)

    # Response 200 schema references QueryResponse
    resp_schema_ref = query_op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert "QueryResponse" in resp_schema_ref
    resp_model = openapi["components"]["schemas"]["QueryResponse"]
    for expected_field in [
        "question",
        "answer",
        "abstained",
        "retrieved_count",
        "sources",
    ]:
        assert expected_field in resp_model["required"]


# ==============================================================================
# 7. Route Tests for Generated Citation Validation (Correction 1)
# ==============================================================================


def test_post_query_500_invalid_citation_marker(query_client) -> None:
    """Verify endpoint returns HTTP 500 when model generates an invalid citation marker."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    # Seed with chunk so retrieval succeeds and calls generator
    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Port is 8080.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    # Generator produces invalid citation [S99]
    generator.canned_answer = "The server port is 8080. [S99]"

    response = client.post("/api/v1/query", json={"question": "What is the port?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Generation service returned an invalid or empty response"


def test_post_query_500_missing_citation_in_normal_answer(query_client) -> None:
    """Verify endpoint returns HTTP 500 when model generates answer with no citations."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Port is 8080.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    # Generator produces normal answer without any citation
    generator.canned_answer = "The server port is 8080."

    response = client.post("/api/v1/query", json={"question": "What is the port?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Generation service returned an invalid or empty response"


def test_post_query_200_canonical_abstention_from_generator_succeeds(query_client) -> None:
    """Verify endpoint returns HTTP 200 with abstained=True when model returns exact canonical abstention."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Marginally related text.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    # Generator returns exact canonical abstention answer without citations
    generator.canned_answer = CANONICAL_ABSTENTION_ANSWER

    response = client.post("/api/v1/query", json={"question": "Query that model decides to abstain on"})
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is True
    assert data["answer"] == CANONICAL_ABSTENTION_ANSWER
    assert data["retrieved_count"] == 0
    assert data["sources"] == []


def test_post_query_200_multiple_valid_citations(query_client) -> None:
    """Verify endpoint returns HTTP 200 preserving answer when model cites multiple valid sources."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunks = [
        ChunkRecord(
            document_id="doc1",
            chunk_index=0,
            text="Port is 8080.",
            embedding=[0.1, 0.1, 0.1, 0.1],
            filename="quickstart.md",
            file_type="md",
        ),
        ChunkRecord(
            document_id="doc2",
            chunk_index=1,
            text="Timeout is 30.",
            embedding=[0.1, 0.1, 0.1, 0.12],
            filename="troubleshooting.md",
            file_type="md",
        ),
    ]
    chroma.add_chunks(chunks)

    generator.canned_answer = "The server port is 8080 [S1], and the timeout is 30 [S2]."

    response = client.post("/api/v1/query", json={"question": "Configuration?"})
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert data["answer"] == "The server port is 8080 [S1], and the timeout is 30 [S2]."
    assert data["retrieved_count"] == 2
    assert len(data["sources"]) == 2


def test_post_query_500_valid_citation_combined_with_malformed_c1(query_client) -> None:
    """Verify endpoint returns HTTP 500 when answer contains a valid citation and a malformed [C1]."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Port is 8080.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    generator.canned_answer = "The server runs on port 8080 [S1]. Additional evidence: [C1]."

    response = client.post("/api/v1/query", json={"question": "What is the port?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Generation service returned an invalid or empty response"


def test_post_query_200_ordinary_bracketed_text_with_valid_citation(query_client) -> None:
    """Verify endpoint returns HTTP 200 allowing ordinary non-citation bracketed text."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Port is 8080.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    generator.canned_answer = "In [production] mode, the server runs on port 8080 [S1]."

    response = client.post("/api/v1/query", json={"question": "What is the port?"})
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert data["answer"] == "In [production] mode, the server runs on port 8080 [S1]."
    assert data["retrieved_count"] == 1
    assert len(data["sources"]) == 1


def test_post_query_500_valid_citation_combined_with_unknown_linked_citation(query_client) -> None:
    """Verify endpoint returns HTTP 500 when answer contains valid citation and unknown linked citation [S99](url)."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Port is 8080.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    generator.canned_answer = "The server runs on port 8080 [S1]. Additional details at [S99](https://example.com)."

    response = client.post("/api/v1/query", json={"question": "What is the port?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Generation service returned an invalid or empty response"


def test_post_query_200_valid_citation_combined_with_ordinary_markdown_link(query_client) -> None:
    """Verify endpoint returns HTTP 200 when answer contains valid citation and ordinary Markdown link."""
    client, query_svc, chroma, embedder, generator, settings = query_client

    chunk = ChunkRecord(
        document_id="doc1",
        chunk_index=0,
        text="Port is 8080.",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="quickstart.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    generator.canned_answer = "Read the [documentation](https://example.com). The port is 8080 [S1]."

    response = client.post("/api/v1/query", json={"question": "What is the port?"})
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert data["answer"] == "Read the [documentation](https://example.com). The port is 8080 [S1]."
    assert data["retrieved_count"] == 1
    assert len(data["sources"]) == 1



