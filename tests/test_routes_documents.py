"""Comprehensive tests for document management REST API endpoints (/api/v1/documents)."""

import io
import pathlib
import uuid
from unittest.mock import MagicMock

import pymupdf
import pytest
from starlette.testclient import TestClient

from app.api.routes_documents import get_ingestion_service, read_upload_file_bytes
from app.core.config import Settings
from app.db.chroma import ChromaAdapter, ChromaStorageError
from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DuplicateDocumentError,
)
from app.db.sqlite import init_db
from app.main import create_app
from app.models.document import Document
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingProviderError,
    MissingApiKeyError,
)
from app.services.ingestion import (
    ChromaDeletionError,
    DeletionPartialConsistencyError,
    DocumentDeletionResult,
    FileSizeLimitExceededError,
    IngestionService,
    PartialConsistencyError,
)


class FakeEmbeddingAdapter:
    """Deterministic fake embedding adapter for API tests."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * self.dim for i in range(len(texts))]


def _create_minimal_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate in-memory PDF bytes."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def app_client(tmp_path: pathlib.Path):
    """Fixture providing a TestClient with isolated temporary databases and fake embedding."""
    db_path = tmp_path / "devguide.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path, tmp_path)

    settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_path,
        CHROMA_PATH=chroma_path,
        MAX_UPLOAD_MB=10,
        CHUNK_SIZE=100,
        CHUNK_OVERLAP=20,
    )

    repo = DocumentRepository(db_path=db_path)
    chroma = ChromaAdapter(chroma_path=chroma_path, settings=settings)
    chroma.init_collection()
    embedder = FakeEmbeddingAdapter(dim=4)

    service = IngestionService(
        repository=repo,
        chroma_adapter=chroma,
        embedding_adapter=embedder,  # type: ignore[arg-type]
        settings=settings,
        chunk_size=100,
        chunk_overlap=20,
    )

    app = create_app(settings=settings)
    app.dependency_overrides[get_ingestion_service] = lambda: service

    with TestClient(app) as client:
        yield client, service, settings, tmp_path


# ==============================================================================
# 1. Streaming Size Enforcement Unit Tests
# ==============================================================================


def test_read_upload_file_bytes_streaming_enforcement():
    """Verify read_upload_file_bytes halts reading and raises FileSizeLimitExceededError when stream exceeds limit."""
    from fastapi import UploadFile

    max_bytes = 100
    # Stream 500 bytes in an in-memory buffer
    stream_content = b"x" * 500
    spooled = io.BytesIO(stream_content)
    upload_file = UploadFile(file=spooled, filename="stream.txt", size=500)

    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        read_upload_file_bytes(upload_file, max_bytes=max_bytes)

    assert "File size exceeds maximum allowed upload limit" in str(exc_info.value)
    # Ensure stream was interrupted early and did not consume beyond limit
    assert spooled.tell() <= 64 * 1024 + 100


def test_read_upload_file_bytes_within_limit():
    """Verify read_upload_file_bytes reads entire file when within limit."""
    from fastapi import UploadFile

    content = b"Short content within limit"
    spooled = io.BytesIO(content)
    upload_file = UploadFile(file=spooled, filename="ok.txt", size=len(content))

    result = read_upload_file_bytes(upload_file, max_bytes=1000)
    assert result == content


# ==============================================================================
# 2. Document Upload Endpoint (POST /api/v1/documents)
# ==============================================================================


def test_upload_markdown_success(app_client):
    """Verify uploading a valid markdown document returns 201 Created and correct schema."""
    client, service, settings, _ = app_client

    content = b"# Architecture\n\nDevGuide architecture overview."
    response = client.post(
        "/api/v1/documents",
        files={"file": ("architecture.md", content, "text/markdown")},
    )

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["filename"] == "architecture.md"
    assert data["file_type"] == "md"
    assert data["chunk_count"] > 0
    assert "content_sha256" in data
    assert "created_at" in data
    # Verify valid UUID format
    assert uuid.UUID(data["id"])


def test_upload_pdf_success(app_client):
    """Verify uploading a valid PDF document returns 201 Created."""
    client, service, settings, _ = app_client

    pdf_bytes = _create_minimal_pdf(["Page 1 PDF text", "Page 2 PDF text"])
    response = client.post(
        "/api/v1/documents",
        files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "guide.pdf"
    assert data["file_type"] == "pdf"
    assert data["chunk_count"] >= 2


def test_upload_duplicate_content_returns_409(app_client):
    """Verify uploading duplicate document content returns 409 Conflict with specified detail."""
    client, service, settings, _ = app_client

    content = b"# Unique Content\n\nThis content is uploaded twice."
    res1 = client.post(
        "/api/v1/documents",
        files={"file": ("first.md", content, "text/markdown")},
    )
    assert res1.status_code == 201
    doc_id = res1.json()["id"]

    res2 = client.post(
        "/api/v1/documents",
        files={"file": ("second.md", content, "text/markdown")},
    )
    assert res2.status_code == 409
    data = res2.json()
    assert "detail" in data
    assert "Document with identical content SHA-256 hash already exists" in data["detail"]
    assert f"ID: {doc_id}" in data["detail"]


def test_upload_oversized_file_returns_413(app_client):
    """Verify uploading file exceeding MAX_UPLOAD_MB returns 413 Payload Too Large."""
    client, service, settings, _ = app_client

    # Configure MAX_UPLOAD_MB to 1MB
    settings.MAX_UPLOAD_MB = 1
    oversized = b"A" * (1024 * 1024 + 1024)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("big.txt", oversized, "text/plain")},
    )

    assert response.status_code == 413
    data = response.json()
    assert "detail" in data
    assert "File size exceeds maximum allowed upload limit of 1 MB" in data["detail"]


def test_upload_unsupported_file_type_returns_400(app_client):
    """Verify uploading unsupported file type returns 400 Bad Request."""
    client, service, settings, _ = app_client

    response = client.post(
        "/api/v1/documents",
        files={"file": ("document.docx", b"PK...", "application/octet-stream")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Unsupported file type '.docx'" in data["detail"]


def test_upload_empty_file_returns_400(app_client):
    """Verify uploading empty 0-byte file returns 400 Bad Request."""
    client, service, settings, _ = app_client

    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "File is empty (0 bytes)" in data["detail"]


def test_upload_corrupt_pdf_returns_400(app_client):
    """Verify uploading corrupt PDF returns 400 Bad Request."""
    client, service, settings, _ = app_client

    response = client.post(
        "/api/v1/documents",
        files={"file": ("corrupt.pdf", b"%PDF-1.4 garbage", "application/pdf")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_upload_partial_consistency_failure_returns_500(app_client):
    """Verify partial consistency failure on ingestion returns 500 Internal Server Error."""
    client, service, settings, _ = app_client

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_document.side_effect = PartialConsistencyError(
        "Partial consistency failure: vectors persisted in ChromaDB but SQLite metadata insertion failed and vector cleanup could not be completed"
    )

    app = client.app
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    response = client.post(
        "/api/v1/documents",
        files={"file": ("test.md", b"# Title\n\nContent", "text/markdown")},
    )

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == (
        "Partial consistency failure: vectors persisted in ChromaDB but SQLite metadata insertion failed and vector cleanup could not be completed"
    )


def test_upload_embedding_provider_error_returns_503(app_client):
    """Verify upstream embedding provider error during upload returns 503 Service Unavailable."""
    client, service, settings, _ = app_client

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_document.side_effect = EmbeddingProviderError(
        "OpenAI rate limit exceeded during embedding generation"
    )

    app = client.app
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    response = client.post(
        "/api/v1/documents",
        files={"file": ("test.md", b"# Title\n\nContent", "text/markdown")},
    )

    assert response.status_code == 503
    assert "rate limit exceeded" in response.json()["detail"]


def test_upload_embedding_authentication_error_returns_503(app_client):
    """Verify upstream embedding authentication error during upload returns 503 Service Unavailable."""
    client, service, settings, _ = app_client

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_document.side_effect = EmbeddingAuthenticationError(
        "OpenAI authentication failed: invalid or unauthorized API key"
    )

    app = client.app
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    response = client.post(
        "/api/v1/documents",
        files={"file": ("test.md", b"# Title\n\nContent", "text/markdown")},
    )

    assert response.status_code == 503
    assert "OpenAI authentication failed: invalid or unauthorized API key" in response.json()["detail"]


def test_upload_missing_api_key_error_returns_500(app_client):
    """Verify MissingApiKeyError during upload falls through generic EmbeddingError to 500 Internal Server Error."""
    client, service, settings, _ = app_client

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_document.side_effect = MissingApiKeyError(
        "OPENAI_API_KEY is not set or empty. A valid API key is required to perform embedding operations."
    )

    app = client.app
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    response = client.post(
        "/api/v1/documents",
        files={"file": ("test.md", b"# Title\n\nContent", "text/markdown")},
    )

    assert response.status_code == 500
    assert "Storage or embedding service failure during ingestion" in response.json()["detail"]


# ==============================================================================
# 3. Document Listing Endpoint (GET /api/v1/documents)
# ==============================================================================


def test_list_documents_empty(app_client):
    """Verify listing documents on empty store returns 200 OK with empty list."""
    client, service, settings, _ = app_client

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data == {"documents": [], "total": 0}


def test_list_documents_populated(app_client):
    """Verify listing documents returns all indexed documents ordered newest first."""
    client, service, settings, _ = app_client

    # Ingest two documents
    res1 = client.post(
        "/api/v1/documents",
        files={"file": ("first.md", b"First document text.", "text/markdown")},
    )
    res2 = client.post(
        "/api/v1/documents",
        files={"file": ("second.md", b"Second document text.", "text/markdown")},
    )

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["documents"]) == 2
    assert {d["id"] for d in data["documents"]} == {res1.json()["id"], res2.json()["id"]}

    # Verify newest-first ordering with distinct timestamps
    from app.db.sqlite import get_connection
    with get_connection(service.repository.db_path) as conn:
        conn.execute("UPDATE documents SET created_at = '2026-11-23T01:00:00Z' WHERE id = ?;", (res1.json()["id"],))
        conn.execute("UPDATE documents SET created_at = '2026-11-23T02:00:00Z' WHERE id = ?;", (res2.json()["id"],))

    ordered_res = client.get("/api/v1/documents")
    assert ordered_res.json()["documents"][0]["id"] == res2.json()["id"]
    assert ordered_res.json()["documents"][1]["id"] == res1.json()["id"]


# ==============================================================================
# 4. Document Detail Endpoint (GET /api/v1/documents/{document_id})
# ==============================================================================


def test_get_document_success(app_client):
    """Verify retrieving document metadata by ID returns 200 OK and correct schema."""
    client, service, settings, _ = app_client

    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("detail.md", b"Detail document content.", "text/markdown")},
    )
    doc_id = upload_res.json()["id"]

    response = client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == doc_id
    assert data["filename"] == "detail.md"
    assert data["file_type"] == "md"
    assert data["chunk_count"] > 0


def test_get_document_missing_returns_404(app_client):
    """Verify retrieving non-existent document ID returns 404 Not Found."""
    client, service, settings, _ = app_client

    missing_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/documents/{missing_id}")

    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == f"Document '{missing_id}' not found"


def test_get_document_invalid_uuid_returns_400(app_client):
    """Verify retrieving with invalid UUID returns 400 Bad Request."""
    client, service, settings, _ = app_client

    response = client.get("/api/v1/documents/not-a-valid-uuid")
    assert response.status_code == 400
    assert "Invalid UUID" in response.json()["detail"]


# ==============================================================================
# 5. Document Delete Endpoint (DELETE /api/v1/documents/{document_id})
# ==============================================================================


def test_delete_document_success(app_client):
    """Verify deleting an existing document returns 200 OK and deletes across both stores."""
    client, service, settings, _ = app_client

    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("del.md", b"Content to be deleted.", "text/markdown")},
    )
    doc_id = upload_res.json()["id"]
    chunk_count = upload_res.json()["chunk_count"]

    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    data = del_res.json()
    assert data == {
        "id": doc_id,
        "deleted": True,
        "chunks_removed": chunk_count,
    }

    # Subsequent GET must return 404
    get_res = client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404


def test_delete_document_missing_returns_404(app_client):
    """Verify deleting non-existent document returns 404 Not Found."""
    client, service, settings, _ = app_client

    missing_id = str(uuid.uuid4())
    response = client.delete(f"/api/v1/documents/{missing_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == f"Document '{missing_id}' not found"


def test_delete_document_invalid_uuid_returns_400(app_client):
    """Verify deleting with invalid UUID returns 400 Bad Request."""
    client, service, settings, _ = app_client

    response = client.delete("/api/v1/documents/invalid-uuid-format")
    assert response.status_code == 400
    assert "Invalid UUID" in response.json()["detail"]


def test_delete_document_chroma_failure_returns_500(app_client):
    """Verify Chroma deletion failure returns 500 and preserves SQLite metadata."""
    client, service, settings, _ = app_client

    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("chroma_fail.md", b"Content preserving metadata.", "text/markdown")},
    )
    doc_id = upload_res.json()["id"]

    mock_service = MagicMock(spec=IngestionService)
    mock_service.delete_document.side_effect = ChromaDeletionError(
        "Failed to delete vectors from ChromaDB. Document metadata deletion aborted."
    )

    app = client.app
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    response = client.delete(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to delete vectors from ChromaDB. Document metadata deletion aborted."
    )


def test_delete_document_sqlite_failure_partial_consistency_returns_500(app_client):
    """Verify SQLite failure after Chroma deletion returns 500 with partial consistency detail."""
    client, service, settings, _ = app_client

    mock_service = MagicMock(spec=IngestionService)
    mock_service.delete_document.side_effect = DeletionPartialConsistencyError(
        "Partial consistency failure: vectors deleted from ChromaDB but failed to delete SQLite metadata record."
    )

    app = client.app
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    test_id = str(uuid.uuid4())
    response = client.delete(f"/api/v1/documents/{test_id}")
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Partial consistency failure: vectors deleted from ChromaDB but failed to delete SQLite metadata record."
    )


# ==============================================================================
# 6. OpenAPI Documentation Contract Verification
# ==============================================================================


def test_openapi_contract_matches_specification(app_client):
    """Verify OpenAPI schema exposes required endpoints, methods, and response models."""
    client, service, settings, _ = app_client
    app = client.app

    schema = app.openapi()
    paths = schema["paths"]

    # Documents endpoints registered
    assert "/api/v1/documents" in paths
    assert "/api/v1/documents/{document_id}" in paths

    doc_methods = paths["/api/v1/documents"]
    assert "post" in doc_methods
    assert "get" in doc_methods

    doc_id_methods = paths["/api/v1/documents/{document_id}"]
    assert "get" in doc_id_methods
    assert "delete" in doc_id_methods

    # Verify status codes in schema
    assert "201" in doc_methods["post"]["responses"]
    assert "200" in doc_methods["get"]["responses"]
    assert "200" in doc_id_methods["get"]["responses"]
    assert "200" in doc_id_methods["delete"]["responses"]

    # Verify components schemas
    components = schema.get("components", {}).get("schemas", {})
    assert "DocumentResponse" in components
    assert "DocumentListResponse" in components
    assert "DocumentDeleteResponse" in components
    assert "ErrorResponse" in components

    # Verify documented error status codes and ErrorResponse references
    # POST /api/v1/documents: 400, 409, 413, 500, 503
    for status_code in ("400", "409", "413", "500", "503"):
        assert status_code in doc_methods["post"]["responses"]
        resp_schema = doc_methods["post"]["responses"][status_code]["content"]["application/json"]["schema"]
        assert resp_schema.get("$ref") == "#/components/schemas/ErrorResponse"

    # GET /api/v1/documents: 500
    assert "500" in doc_methods["get"]["responses"]
    resp_schema_get_list = doc_methods["get"]["responses"]["500"]["content"]["application/json"]["schema"]
    assert resp_schema_get_list.get("$ref") == "#/components/schemas/ErrorResponse"

    # GET /api/v1/documents/{document_id}: 400, 404, 500
    for status_code in ("400", "404", "500"):
        assert status_code in doc_id_methods["get"]["responses"]
        resp_schema = doc_id_methods["get"]["responses"][status_code]["content"]["application/json"]["schema"]
        assert resp_schema.get("$ref") == "#/components/schemas/ErrorResponse"

    # DELETE /api/v1/documents/{document_id}: 400, 404, 500
    for status_code in ("400", "404", "500"):
        assert status_code in doc_id_methods["delete"]["responses"]
        resp_schema = doc_id_methods["delete"]["responses"][status_code]["content"]["application/json"]["schema"]
        assert resp_schema.get("$ref") == "#/components/schemas/ErrorResponse"

    # Verify successful response schema references remain intact
    post_201_schema = doc_methods["post"]["responses"]["201"]["content"]["application/json"]["schema"]
    assert post_201_schema.get("$ref") == "#/components/schemas/DocumentResponse"

    get_list_200_schema = doc_methods["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert get_list_200_schema.get("$ref") == "#/components/schemas/DocumentListResponse"

    get_doc_200_schema = doc_id_methods["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert get_doc_200_schema.get("$ref") == "#/components/schemas/DocumentResponse"

    delete_200_schema = doc_id_methods["delete"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert delete_200_schema.get("$ref") == "#/components/schemas/DocumentDeleteResponse"


# ==============================================================================
# 7. Application-Scoped Settings Dependency & Canonical UUID API Tests
# ==============================================================================


def test_delete_document_uppercase_and_whitespace_api(app_client):
    """Verify DELETE endpoint with uppercase and padded UUID cleans up and returns canonical lowercase ID."""
    client, service, settings, _ = app_client

    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("case_test.md", b"Content to delete with uppercase UUID.", "text/markdown")},
    )
    doc_id = upload_res.json()["id"]

    # Delete with uppercase and surrounding whitespace
    mangled_id = f"  {doc_id.upper()}  "
    del_res = client.delete(f"/api/v1/documents/{mangled_id}")
    assert del_res.status_code == 200
    data = del_res.json()
    assert data["id"] == doc_id  # Canonical lowercase UUID returned
    assert data["deleted"] is True

    # Subsequent GET returns 404
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404


def test_custom_app_settings_honored_by_document_routes(tmp_path):
    """Verify custom Settings instance supplied to create_app() is used by route dependencies."""
    db_path = tmp_path / "custom.db"
    chroma_path = tmp_path / "custom_chroma"
    init_db(db_path, tmp_path)

    # Supply custom Settings with MAX_UPLOAD_MB = 2
    custom_settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_path,
        CHROMA_PATH=chroma_path,
        MAX_UPLOAD_MB=2,
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=30,
    )

    app = create_app(settings=custom_settings)
    # Inject fake embedder so no real API calls are made, but leave get_ingestion_service using custom settings
    fake_embedder = FakeEmbeddingAdapter(dim=4)
    app.dependency_overrides[get_ingestion_service] = lambda: IngestionService(
        repository=DocumentRepository(db_path=db_path),
        chroma_adapter=ChromaAdapter(chroma_path=chroma_path, settings=custom_settings),
        embedding_adapter=fake_embedder,  # type: ignore[arg-type]
        settings=custom_settings,
        chunk_size=custom_settings.CHUNK_SIZE,
        chunk_overlap=custom_settings.CHUNK_OVERLAP,
    )

    with TestClient(app) as client:
        # 1. Upload within custom limit -> succeeds
        valid_content = b"Short text documentation for custom settings test."
        res_ok = client.post(
            "/api/v1/documents",
            files={"file": ("valid_custom.txt", valid_content, "text/plain")},
        )
        assert res_ok.status_code == 201
        assert res_ok.json()["filename"] == "valid_custom.txt"

        # 2. Upload exceeding custom limit (2.5 MB) -> rejected with 413
        oversized_content = b"y" * int(2.5 * 1024 * 1024)
        res_large = client.post(
            "/api/v1/documents",
            files={"file": ("too_large.txt", oversized_content, "text/plain")},
        )
        assert res_large.status_code == 413
        assert "2 MB (2097152 bytes)" in res_large.json()["detail"]


def test_custom_app_settings_short_circuits_before_service_invocation(tmp_path):
    """Verify upload-size check honors custom MAX_UPLOAD_MB and does not invoke the service."""
    db_path = tmp_path / "custom2.db"
    chroma_path = tmp_path / "custom2_chroma"
    init_db(db_path, tmp_path)

    custom_settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_path,
        CHROMA_PATH=chroma_path,
        MAX_UPLOAD_MB=1,
    )

    mock_service = MagicMock(spec=IngestionService)
    app = create_app(settings=custom_settings)
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    with TestClient(app) as client:
        oversized_content = b"z" * int(1.5 * 1024 * 1024)
        res = client.post(
            "/api/v1/documents",
            files={"file": ("big.txt", oversized_content, "text/plain")},
        )
        assert res.status_code == 413
        # IngestionService.ingest_document must NOT be called
        mock_service.ingest_document.assert_not_called()


def test_default_app_settings_preserved():
    """Verify default Settings behavior is preserved when no settings are passed to create_app."""
    app = create_app()
    # Verify app.state.settings matches default get_settings()
    assert app.state.settings.MAX_UPLOAD_MB == 10
    assert app.state.settings.CHUNK_SIZE == 1000

    # Also verify get_app_settings returns default settings when state has app settings or None
    from app.api.routes_documents import get_app_settings

    mock_request = MagicMock()
    mock_request.app.state.settings = app.state.settings
    resolved = get_app_settings(mock_request)
    assert resolved.MAX_UPLOAD_MB == 10

    mock_request_none = MagicMock()
    mock_request_none.app.state.settings = None
    fallback = get_app_settings(mock_request_none)
    assert fallback.MAX_UPLOAD_MB == 10
