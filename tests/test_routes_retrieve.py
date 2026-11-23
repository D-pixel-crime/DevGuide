"""Comprehensive tests for semantic retrieval REST API endpoint (/api/v1/retrieve)."""

import pathlib
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from app.api.routes_retrieve import get_retrieval_service
from app.core.config import Settings
from app.db.chroma import ChromaAdapter, ChromaStorageError, ChunkRecord, VectorSearchResult
from app.db.sqlite import init_db
from app.main import create_app
from app.models.schemas import RetrieveMatch, RetrieveRequest, RetrieveResponse
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    MissingApiKeyError,
)
from app.services.retrieval import (
    InvalidQueryError,
    RetrievalResult,
    RetrievalService,
    RetrievedMatch,
)


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


@pytest.fixture
def app_client(tmp_path: pathlib.Path):
    """Fixture providing TestClient with isolated temporary databases and fake embedding."""
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
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
    )

    chroma = ChromaAdapter(chroma_path=chroma_path, settings=settings)
    chroma.init_collection()
    embedder = FakeEmbeddingAdapter(dim=4)

    service = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=embedder,
        settings=settings,
    )

    app = create_app(settings=settings)
    app.dependency_overrides[get_retrieval_service] = lambda: service

    with TestClient(app) as client:
        yield client, service, chroma, embedder, settings


# ==============================================================================
# 1. Successful Retrieval Tests (200 OK)
# ==============================================================================


def test_post_retrieve_200_with_matches(app_client) -> None:
    """Verify POST /api/v1/retrieve returns 200 with canonical matches and response schema."""
    client, service, chroma, embedder, settings = app_client

    # Add chunks directly to ChromaDB
    chunks = [
        ChunkRecord(
            document_id="f8a12c34-90ab-4cde-8123-112233445566",
            chunk_index=0,
            text="To resolve database locked errors, increase TASKBOARD_DB_TIMEOUT=30 in your .env configuration file.",
            embedding=[0.1, 0.1, 0.1, 0.1],  # Identical to query vector -> distance ~ 0.0
            filename="troubleshooting.md",
            file_type="md",
            page_number=None,
        ),
        ChunkRecord(
            document_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            chunk_index=1,
            text="Physical PDF chunk content regarding database settings.",
            embedding=[0.1, 0.1, 0.1, 0.12],  # Close vector -> distance < 0.70
            filename="manual.pdf",
            file_type="pdf",
            page_number=3,
        ),
    ]
    chroma.add_chunks(chunks)

    payload = {
        "question": "How do I configure database timeout?",
        "top_k": 5,
    }
    response = client.post("/api/v1/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Canonical top-level fields
    assert data["question"] == "How do I configure database timeout?"
    assert data["requested_top_k"] == 5
    assert data["has_relevant_context"] is True
    assert data["retrieved_count"] == 2
    assert len(data["matches"]) == 2

    # 2. Strict rejection of obsolete fields
    assert "query" not in data
    assert "results" not in data
    assert "total_retrieved" not in data

    # 3. Canonical match fields
    m0 = data["matches"][0]
    assert m0["text"] == "To resolve database locked errors, increase TASKBOARD_DB_TIMEOUT=30 in your .env configuration file."
    assert m0["document_id"] == "f8a12c34-90ab-4cde-8123-112233445566"
    assert m0["filename"] == "troubleshooting.md"
    assert m0["file_type"] == "md"
    assert m0["chunk_index"] == 0
    assert m0["page_number"] is None  # null in JSON for non-PDF
    assert isinstance(m0["distance"], float)
    assert m0["distance"] <= 0.70
    assert "source_label" not in m0

    m1 = data["matches"][1]
    assert m1["filename"] == "manual.pdf"
    assert m1["file_type"] == "pdf"
    assert m1["page_number"] == 3  # integer in JSON for PDF
    assert "source_label" not in m1


def test_post_retrieve_200_default_top_k(app_client) -> None:
    """Verify that omitting top_k defaults to DEFAULT_TOP_K (5)."""
    client, service, chroma, embedder, settings = app_client

    response = client.post(
        "/api/v1/retrieve",
        json={"question": "What is the timeout?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["requested_top_k"] == 5


def test_post_retrieve_200_explicit_null_top_k(app_client) -> None:
    """Verify explicit null top_k still uses DEFAULT_TOP_K (5)."""
    client, service, chroma, embedder, settings = app_client

    response = client.post(
        "/api/v1/retrieve",
        json={"question": "What is the timeout?", "top_k": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["requested_top_k"] == 5


def test_post_retrieve_200_custom_top_k(app_client) -> None:
    """Verify that explicit top_k is reflected in requested_top_k."""
    client, service, chroma, embedder, settings = app_client

    response = client.post(
        "/api/v1/retrieve",
        json={"question": "What is the timeout?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["requested_top_k"] == 3


def test_post_retrieve_200_empty_collection(app_client) -> None:
    """Verify querying an empty store returns has_relevant_context=False and empty matches list."""
    client, service, chroma, embedder, settings = app_client

    response = client.post(
        "/api/v1/retrieve",
        json={"question": "What is the automated database backup schedule?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_relevant_context"] is False
    assert data["retrieved_count"] == 0
    assert data["matches"] == []
    assert data["requested_top_k"] == 5


def test_post_retrieve_200_zero_relevant_matches(app_client) -> None:
    """Verify chunks with distance > 0.70 are excluded and return empty matches."""
    client, service, chroma, embedder, settings = app_client

    # Add chunk with orthogonal vector: query is [0.1, 0.1, 0.1, 0.1], chunk is [-0.1, -0.1, -0.1, -0.1]
    # In cosine space, opposite vectors have distance ~ 2.0 (> 0.70)
    chunk = ChunkRecord(
        document_id="doc-distant",
        chunk_index=0,
        text="Completely opposite topic chunk.",
        embedding=[-0.1, -0.1, -0.1, -0.1],
        filename="unrelated.txt",
        file_type="txt",
    )
    chroma.add_chunks([chunk])

    response = client.post(
        "/api/v1/retrieve",
        json={"question": "What is the automated database backup schedule?", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_relevant_context"] is False
    assert data["retrieved_count"] == 0
    assert data["matches"] == []


# ==============================================================================
# 2. Validation Error Tests (400 Bad Request & 422 Unprocessable Entity)
# ==============================================================================


def test_post_retrieve_400_empty_question(app_client) -> None:
    """Verify empty question string returns 400 Bad Request."""
    client, _, _, _, _ = app_client

    response = client.post("/api/v1/retrieve", json={"question": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_post_retrieve_400_whitespace_question(app_client) -> None:
    """Verify whitespace-only question string returns 400 Bad Request."""
    client, _, _, _, _ = app_client

    response = client.post("/api/v1/retrieve", json={"question": "   \n\t  "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_post_retrieve_400_question_too_long(app_client) -> None:
    """Verify question exceeding 1000 characters returns 400 Bad Request."""
    client, _, _, _, _ = app_client

    response = client.post("/api/v1/retrieve", json={"question": "x" * 1001})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot exceed 1000 characters"


def test_post_retrieve_400_question_exceeds_length_with_whitespace(app_client) -> None:
    """Verify question exceeding 1000 characters before trimming returns 400 Bad Request."""
    client, _, _, _, _ = app_client

    # 200 leading spaces + 10-char word + 800 trailing spaces = 1010 characters (>1000)
    # Trimmed content is only 10 characters (<1000)
    padded_question = (" " * 200) + "How to run" + (" " * 800)
    assert len(padded_question) == 1010
    assert len(padded_question.strip()) == 10

    response = client.post("/api/v1/retrieve", json={"question": padded_question})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot exceed 1000 characters"


def test_post_retrieve_400_top_k_below_min(app_client) -> None:
    """Verify top_k < 1 returns 400 Bad Request matching docs/API.md."""
    client, _, _, _, _ = app_client

    response = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": 0})
    assert response.status_code == 400
    assert response.json()["detail"] == "Parameter 'top_k' must be between 1 and 10"


def test_post_retrieve_400_top_k_above_max(app_client) -> None:
    """Verify top_k > 10 returns 400 Bad Request matching docs/API.md."""
    client, _, _, _, _ = app_client

    response = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": 11})
    assert response.status_code == 400
    assert response.json()["detail"] == "Parameter 'top_k' must be between 1 and 10"


def test_post_retrieve_422_top_k_boolean(app_client) -> None:
    """Verify boolean top_k is rejected by request validation with 422 Unprocessable Entity."""
    client, _, _, _, _ = app_client

    response_true = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": True})
    assert response_true.status_code == 422

    response_false = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": False})
    assert response_false.status_code == 422


def test_post_retrieve_422_top_k_invalid_types(app_client) -> None:
    """Verify string, float, and list top_k values return 422 Unprocessable Entity."""
    client, _, _, _, _ = app_client

    # string
    res_str = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": "5"})
    assert res_str.status_code == 422

    # float
    res_float = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": 5.5})
    assert res_float.status_code == 422

    # list
    res_list = client.post("/api/v1/retrieve", json={"question": "Valid question", "top_k": [5]})
    assert res_list.status_code == 422


def test_post_retrieve_422_missing_question_field(app_client) -> None:
    """Verify missing required question field returns 422 Unprocessable Entity."""
    client, _, _, _, _ = app_client

    response = client.post("/api/v1/retrieve", json={})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_post_retrieve_422_invalid_json_body(app_client) -> None:
    """Verify non-dictionary JSON body returns 422 Unprocessable Entity."""
    client, _, _, _, _ = app_client

    response = client.post(
        "/api/v1/retrieve",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ==============================================================================
# 3. Upstream Dependency & Storage Error Tests (500 & 503)
# ==============================================================================


def test_post_retrieve_503_upstream_openai_error(app_client) -> None:
    """Verify upstream OpenAI provider errors return 503 Service Unavailable."""
    client, service, _, _, _ = app_client

    mock_service = MagicMock(spec=RetrievalService)
    mock_service.retrieve.side_effect = EmbeddingProviderError("OpenAI API connection timeout")

    app = client.app
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    response = client.post("/api/v1/retrieve", json={"question": "Query"})
    assert response.status_code == 503
    assert response.json()["detail"] == "OpenAI API connection timeout"


def test_post_retrieve_503_openai_auth_error(app_client) -> None:
    """Verify OpenAI authentication failure returns 503 Service Unavailable."""
    client, service, _, _, _ = app_client

    mock_service = MagicMock(spec=RetrievalService)
    mock_service.retrieve.side_effect = EmbeddingAuthenticationError("Invalid API key")

    app = client.app
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    response = client.post("/api/v1/retrieve", json={"question": "Query"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Invalid API key"


def test_post_retrieve_500_chroma_storage_error(app_client) -> None:
    """Verify internal ChromaDB storage failures return 500 Internal Server Error."""
    client, service, _, _, _ = app_client

    mock_service = MagicMock(spec=RetrievalService)
    mock_service.retrieve.side_effect = ChromaStorageError("ChromaDB disk read failed")

    app = client.app
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    response = client.post("/api/v1/retrieve", json={"question": "Query"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Storage or embedding service failure during retrieval"


# ==============================================================================
# 4. Strict Assertion That Generation Is Never Invoked
# ==============================================================================


def test_post_retrieve_never_invokes_generation(app_client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strictly assert that POST /api/v1/retrieve does not invoke LLM chat or responses API."""
    client, service, chroma, embedder, settings = app_client

    # Add chunk so retrieval produces a match
    chunk = ChunkRecord(
        document_id="doc-gen-test",
        chunk_index=0,
        text="Sample documentation text",
        embedding=[0.1, 0.1, 0.1, 0.1],
        filename="sample.md",
        file_type="md",
    )
    chroma.add_chunks([chunk])

    # Intercept any potential chat completion or generation calls
    generation_spy = MagicMock()
    monkeypatch.setattr("openai.resources.chat.Completions.create", generation_spy, raising=False)

    response = client.post(
        "/api/v1/retrieve",
        json={"question": "Sample query", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["has_relevant_context"] is True

    # Confirm generation API was never called
    assert generation_spy.call_count == 0


# ==============================================================================
# 5. OpenAPI Specification Verification
# ==============================================================================


def test_openapi_contains_retrieve_endpoint(app_client) -> None:
    """Verify that OpenAPI schema defines POST /api/v1/retrieve correctly."""
    client, _, _, _, _ = app_client

    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()

    paths = openapi["paths"]
    assert "/api/v1/retrieve" in paths
    retrieve_op = paths["/api/v1/retrieve"]["post"]

    assert "retrieval" in retrieve_op["tags"]
    assert "200" in retrieve_op["responses"]
    assert "400" in retrieve_op["responses"]
    assert "500" in retrieve_op["responses"]
    assert "503" in retrieve_op["responses"]

    # Request body schema references RetrieveRequest
    req_schema_ref = retrieve_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert "RetrieveRequest" in req_schema_ref
    req_model = openapi["components"]["schemas"]["RetrieveRequest"]

    # 1. question is a required string declaring maxLength=1000
    assert "question" in req_model["required"]
    q_prop = req_model["properties"]["question"]
    assert q_prop["type"] == "string"
    assert q_prop["maxLength"] == 1000

    # 2. top_k is an optional integer declaring minimum=1 and maximum=10
    assert "top_k" not in req_model.get("required", [])
    top_k_prop = req_model["properties"]["top_k"]
    top_k_types = [t.get("type") for t in top_k_prop.get("anyOf", [])]
    assert "integer" in top_k_types or top_k_prop.get("type") == "integer"
    assert top_k_prop["minimum"] in (1, 1.0)
    assert top_k_prop["maximum"] in (10, 10.0)

    # 3. Response 200 schema references RetrieveResponse
    resp_schema_ref = retrieve_op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert "RetrieveResponse" in resp_schema_ref
    resp_model = openapi["components"]["schemas"]["RetrieveResponse"]
    for expected_field in [
        "question",
        "requested_top_k",
        "has_relevant_context",
        "retrieved_count",
        "matches",
    ]:
        assert expected_field in resp_model["required"]
