"""Comprehensive tests for IngestionService orchestration and failure boundaries."""

import pathlib
import sqlite3
import uuid
from typing import Any
from unittest.mock import MagicMock

import pymupdf
import pytest

from app.core.config import Settings
from app.db.chroma import ChromaAdapter, ChromaStorageError, ChunkRecord
from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DuplicateDocumentError,
)
from app.db.sqlite import init_db
from app.models.document import Document
from app.services.embedding import (
    EmbeddingProviderError,
    MissingApiKeyError,
    OpenAIEmbeddingAdapter,
)
from app.services.ingestion import (
    ChromaDeletionError,
    ChunkingError,
    DeletionPartialConsistencyError,
    DocumentDeletionResult,
    EmptyExtractionError,
    EmptyFileError,
    FileSizeLimitExceededError,
    FileValidationError,
    ImageOnlyPDFError,
    IngestionService,
    IngestionStorageError,
    PartialConsistencyError,
    UnsupportedFileTypeError,
)


class FakeEmbeddingAdapter:
    """Deterministic fake embedding adapter for tests without network calls."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.call_count = 0
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.embedded_texts.extend(texts)
        return [[0.05 * (i + 1)] * self.dim for i in range(len(texts))]


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


def _create_image_only_pdf() -> bytes:
    """Helper to generate in-memory PDF containing an image but no extractable text."""
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 50, 200, 200)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), 0)
    page.insert_image(rect, pixmap=pix)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def service_fixture(tmp_path: pathlib.Path):
    """Fixture providing an IngestionService wired to temporary SQLite and ChromaDB."""
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
    return service, repo, chroma, embedder, tmp_path


# ==============================================================================
# 1. Ingestion Success Across File Types & Invariants
# ==============================================================================


def test_ingest_markdown_success(service_fixture):
    """Verify successful ingestion of markdown document."""
    service, repo, chroma, embedder, tmp_path = service_fixture

    content = b"# Introduction\n\nThis is a sample markdown document for DevGuide ingestion."
    doc = service.ingest_document("intro.md", content)

    assert isinstance(doc, Document)
    assert doc.filename == "intro.md"
    assert doc.file_type == "md"
    assert doc.chunk_count > 0
    assert doc.id is not None

    # Verify SQLite record
    fetched = repo.get_document(doc.id)
    assert fetched.id == doc.id
    assert fetched.filename == "intro.md"
    assert fetched.content_sha256 == doc.content_sha256

    # Verify Chroma vector records
    records = chroma.get_records_by_document(doc.id, include_embeddings=True)
    assert len(records) == doc.chunk_count
    for rec in records:
        assert rec.document_id == doc.id
        assert rec.filename == "intro.md"
        assert rec.file_type == "md"
        assert rec.page_number is None
        assert rec.embedding is not None
        assert len(rec.embedding) == 4

    # Verify upload bytes were never persisted to disk
    assert not (tmp_path / "intro.md").exists()


def test_ingest_text_success(service_fixture):
    """Verify successful ingestion of plain text document."""
    service, repo, chroma, embedder, tmp_path = service_fixture

    content = b"Plain text documentation line 1.\nPlain text documentation line 2."
    doc = service.ingest_document("notes.txt", content)

    assert doc.file_type == "txt"
    assert doc.chunk_count > 0
    assert repo.get_document(doc.id) is not None
    records = chroma.get_records_by_document(doc.id)
    assert len(records) == doc.chunk_count
    for rec in records:
        assert rec.page_number is None


def test_ingest_pdf_success_preserves_page_numbers(service_fixture):
    """Verify successful ingestion of PDF preserves 1-based page numbers in Chroma."""
    service, repo, chroma, embedder, tmp_path = service_fixture

    pdf_bytes = _create_minimal_pdf(
        ["Page 1 content text for testing.", "Page 2 content text with more details."]
    )
    doc = service.ingest_document("manual.pdf", pdf_bytes)

    assert doc.file_type == "pdf"
    assert doc.chunk_count >= 2
    records = chroma.get_records_by_document(doc.id)
    assert len(records) == doc.chunk_count
    page_nums = {r.page_number for r in records}
    assert 1 in page_nums
    assert 2 in page_nums


# ==============================================================================
# 2. Duplicate Detection Short-Circuit (Before Expensive Embedding)
# ==============================================================================


def test_duplicate_content_short_circuit(service_fixture):
    """Verify duplicate document is detected before embedding is called."""
    service, repo, chroma, embedder, _ = service_fixture

    content = b"Content to be indexed once and rejected twice."
    first_doc = service.ingest_document("doc1.md", content)
    initial_embed_calls = embedder.call_count
    assert initial_embed_calls > 0

    # Ingest same content under different filename
    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.ingest_document("doc2.md", content)

    err_msg = str(exc_info.value)
    assert "Document with identical content SHA-256 hash already exists" in err_msg
    assert f"ID: {first_doc.id}" in err_msg

    # Crucial assertion: embedding adapter must NOT be invoked again
    assert embedder.call_count == initial_embed_calls


# ==============================================================================
# 3. Stage Failures Prior to Storage
# ==============================================================================


def test_validation_failure_unsupported_extension(service_fixture):
    """Verify unsupported file extension fails immediately without touching storage."""
    service, repo, chroma, embedder, _ = service_fixture
    initial_chroma_count = chroma.count()

    with pytest.raises(UnsupportedFileTypeError):
        service.ingest_document("archive.zip", b"PK\x03\x04...")

    assert embedder.call_count == 0
    assert chroma.count() == initial_chroma_count
    assert len(repo.list_documents()) == 0


def test_validation_failure_empty_file(service_fixture):
    """Verify empty 0-byte file fails validation."""
    service, repo, chroma, embedder, _ = service_fixture

    with pytest.raises(EmptyFileError):
        service.ingest_document("empty.txt", b"")

    assert embedder.call_count == 0
    assert len(repo.list_documents()) == 0


def test_validation_failure_oversized_file(service_fixture):
    """Verify oversized file fails validation."""
    service, repo, chroma, embedder, _ = service_fixture

    oversized = b"a" * (11 * 1024 * 1024)
    with pytest.raises(FileSizeLimitExceededError):
        service.ingest_document("huge.txt", oversized)

    assert embedder.call_count == 0


def test_extraction_failure_image_only_pdf(service_fixture):
    """Verify image-only PDF fails extraction without calling embedder or storage."""
    service, repo, chroma, embedder, _ = service_fixture
    initial_chroma_count = chroma.count()

    pdf_bytes = _create_image_only_pdf()
    with pytest.raises(ImageOnlyPDFError):
        service.ingest_document("scanned.pdf", pdf_bytes)

    assert embedder.call_count == 0
    assert chroma.count() == initial_chroma_count
    assert len(repo.list_documents()) == 0


def test_extraction_failure_empty_text_in_file(service_fixture):
    """Verify file with only whitespace text fails extraction."""
    service, repo, chroma, embedder, _ = service_fixture

    with pytest.raises(EmptyExtractionError):
        service.ingest_document("blank.txt", b"   \n\n\t  ")

    assert embedder.call_count == 0


def test_embedding_failure_aborts_without_storage(service_fixture):
    """Verify embedding failure aborts ingestion before ChromaDB or SQLite writes."""
    service, repo, chroma, _, _ = service_fixture
    initial_chroma_count = chroma.count()

    failing_embedder = MagicMock()
    failing_embedder.embed_texts.side_effect = EmbeddingProviderError("Rate limit exceeded")

    service._embedding_adapter = failing_embedder

    with pytest.raises(EmbeddingProviderError):
        service.ingest_document("test.md", b"# Title\n\nSome test content.")

    assert chroma.count() == initial_chroma_count
    assert len(repo.list_documents()) == 0


# ==============================================================================
# 4. Storage Failure Boundaries & Compensation
# ==============================================================================


def test_chroma_insert_failure_prevents_sqlite_registration(service_fixture):
    """Verify Chroma write failure leaves SQLite completely clean."""
    service, repo, chroma, embedder, _ = service_fixture

    failing_chroma = MagicMock(spec=ChromaAdapter)
    failing_chroma.add_chunks.side_effect = ChromaStorageError("Disk write failed")

    service._chroma_adapter = failing_chroma

    with pytest.raises(ChromaStorageError):
        service.ingest_document("valid.md", b"Valid markdown content.")

    # SQLite must have 0 records
    assert len(repo.list_documents()) == 0


def test_sqlite_failure_triggers_successful_chroma_cleanup(service_fixture):
    """Verify Chroma vectors are cleaned up when SQLite metadata insertion fails."""
    service, repo, chroma, embedder, _ = service_fixture

    # Mock repository.create_document to simulate SQLite failure
    failing_repo = MagicMock(spec=DocumentRepository)
    failing_repo.get_document_by_content_sha256.return_value = None
    failing_repo.create_document.side_effect = sqlite3.OperationalError("database is locked")

    service._repository = failing_repo

    with pytest.raises(IngestionStorageError) as exc_info:
        service.ingest_document("cleanup.md", b"Content that gets cleaned up.")

    assert "Failed to register document metadata in SQLite" in str(exc_info.value)
    # Chroma collection must be empty because compensation succeeded
    assert chroma.count() == 0


def test_sqlite_failure_with_failed_chroma_cleanup_reports_partial_inconsistency(
    service_fixture,
):
    """Verify failure of both SQLite insertion and Chroma cleanup raises PartialConsistencyError."""
    service, repo, chroma, embedder, _ = service_fixture

    failing_repo = MagicMock(spec=DocumentRepository)
    failing_repo.get_document_by_content_sha256.return_value = None
    failing_repo.create_document.side_effect = sqlite3.OperationalError("disk I/O error")

    failing_chroma = MagicMock(spec=ChromaAdapter)
    failing_chroma.add_chunks.return_value = None
    failing_chroma.delete_records_by_document.side_effect = ChromaStorageError("Delete failed")

    service._repository = failing_repo
    service._chroma_adapter = failing_chroma

    with pytest.raises(PartialConsistencyError) as exc_info:
        service.ingest_document("inconsistent.md", b"Content causing inconsistency.")

    err_msg = str(exc_info.value)
    assert "Partial consistency failure: vectors persisted in ChromaDB" in err_msg
    assert "SQLite metadata insertion failed and vector cleanup could not be completed" in err_msg


# ==============================================================================
# 5. Document Deletion Ordering & Boundaries
# ==============================================================================


def test_delete_missing_document_raises_404(service_fixture):
    """Verify deleting non-existent document raises DocumentNotFoundError without Chroma calls."""
    service, repo, chroma, _, _ = service_fixture

    missing_id = str(uuid.uuid4())
    with pytest.raises(DocumentNotFoundError):
        service.delete_document(missing_id)


def test_delete_invalid_uuid_raises_value_error(service_fixture):
    """Verify deleting with invalid UUID raises ValueError."""
    service, repo, chroma, _, _ = service_fixture

    with pytest.raises(ValueError):
        service.delete_document("not-a-valid-uuid")


def test_delete_ordering_chroma_first_failure_preserves_sqlite(service_fixture):
    """Verify if Chroma deletion fails, SQLite metadata deletion is aborted and preserved."""
    service, repo, chroma, _, _ = service_fixture

    doc = service.ingest_document("preserve.md", b"Content to preserve.")
    assert repo.get_document(doc.id) is not None

    # Replace Chroma adapter with one that fails on deletion
    failing_chroma = MagicMock(spec=ChromaAdapter)
    failing_chroma.delete_records_by_document.side_effect = ChromaStorageError("Chroma offline")
    service._chroma_adapter = failing_chroma

    with pytest.raises(ChromaDeletionError) as exc_info:
        service.delete_document(doc.id)

    assert "Failed to delete vectors from ChromaDB" in str(exc_info.value)
    assert "Document metadata deletion aborted." in str(exc_info.value)

    # Document metadata must still be in SQLite!
    assert repo.get_document(doc.id) is not None


def test_delete_chroma_succeeds_but_sqlite_fails_reports_partial_inconsistency(
    service_fixture,
):
    """Verify when Chroma vectors are deleted but SQLite deletion fails, PartialConsistencyError is raised."""
    service, repo, chroma, _, _ = service_fixture

    doc = service.ingest_document("orphan.md", b"Vectors removed, metadata stuck.")

    # Mock repository.delete_document to fail
    failing_repo = MagicMock(spec=DocumentRepository)
    failing_repo.get_document.return_value = doc
    failing_repo.delete_document.side_effect = sqlite3.OperationalError("database is locked")
    service._repository = failing_repo

    with pytest.raises(DeletionPartialConsistencyError) as exc_info:
        service.delete_document(doc.id)

    assert "Partial consistency failure: vectors deleted from ChromaDB but failed to delete SQLite metadata record." in str(exc_info.value)


def test_delete_success_across_both_stores(service_fixture):
    """Verify successful deletion removes both Chroma vectors and SQLite metadata."""
    service, repo, chroma, _, _ = service_fixture

    doc = service.ingest_document("to_delete.md", b"Content to delete cleanly.")
    assert chroma.count() > 0
    assert repo.get_document(doc.id) is not None

    res = service.delete_document(doc.id)
    assert isinstance(res, DocumentDeletionResult)
    assert res.id == doc.id
    assert res.deleted is True
    assert res.chunks_removed == doc.chunk_count

    # Verify document no longer in SQLite
    with pytest.raises(DocumentNotFoundError):
        repo.get_document(doc.id)

    # Verify vectors no longer in Chroma
    assert len(chroma.get_records_by_document(doc.id)) == 0


def test_delete_uppercase_uuid_canonical_cleanup(service_fixture):
    """Verify deleting with uppercase UUID removes vectors, removes SQLite record, and returns canonical lowercase UUID."""
    service, repo, chroma, _, _ = service_fixture

    doc = service.ingest_document("upper.md", b"Content for uppercase deletion test.")
    canonical_id = doc.id
    uppercase_id = canonical_id.upper()
    assert uppercase_id != canonical_id  # Contains hex letters a-f

    # Chroma has records under canonical lowercase ID
    assert len(chroma.get_records_by_document(canonical_id)) > 0
    assert repo.get_document(canonical_id) is not None

    res = service.delete_document(uppercase_id)
    assert res.deleted is True
    assert res.id == canonical_id  # Returns canonical lowercase UUID
    assert res.chunks_removed == doc.chunk_count

    # Verify Chroma vectors removed
    assert len(chroma.get_records_by_document(canonical_id)) == 0
    # Verify SQLite metadata removed
    with pytest.raises(DocumentNotFoundError):
        repo.get_document(canonical_id)


def test_delete_whitespace_padded_uuid_canonical_cleanup(service_fixture):
    """Verify deleting with surrounding whitespace removes vectors, removes SQLite record, and returns canonical lowercase UUID."""
    service, repo, chroma, _, _ = service_fixture

    doc = service.ingest_document("padded.md", b"Content for padded deletion test.")
    canonical_id = doc.id
    padded_id = f"   \t  {canonical_id}  \n "

    assert len(chroma.get_records_by_document(canonical_id)) > 0
    assert repo.get_document(canonical_id) is not None

    res = service.delete_document(padded_id)
    assert res.deleted is True
    assert res.id == canonical_id
    assert res.chunks_removed == doc.chunk_count

    assert len(chroma.get_records_by_document(canonical_id)) == 0
    with pytest.raises(DocumentNotFoundError):
        repo.get_document(canonical_id)


def test_delete_uppercase_and_whitespace_padded_uuid(service_fixture):
    """Verify deleting with combined uppercase and whitespace padding succeeds and returns canonical ID."""
    service, repo, chroma, _, _ = service_fixture

    doc = service.ingest_document("both.md", b"Content for both uppercase and whitespace.")
    canonical_id = doc.id
    mangled_id = f"  {canonical_id.upper()}  "

    res = service.delete_document(mangled_id)
    assert res.deleted is True
    assert res.id == canonical_id
    assert len(chroma.get_records_by_document(canonical_id)) == 0
    with pytest.raises(DocumentNotFoundError):
        repo.get_document(canonical_id)


# ==============================================================================
# 6. Document Listing & Retrieval Service Methods
# ==============================================================================


def test_list_and_get_document_methods(service_fixture):
    """Verify list_documents and get_document service methods."""
    service, repo, _, _, _ = service_fixture

    assert service.list_documents() == []

    doc1 = service.ingest_document("first.md", b"First doc content.")
    doc2 = service.ingest_document("second.md", b"Second doc content.")

    docs = service.list_documents()
    assert len(docs) == 2
    assert {d.id for d in docs} == {doc1.id, doc2.id}

    # Verify newest-first ordering with distinct timestamps
    from app.db.sqlite import get_connection
    with get_connection(repo.db_path) as conn:
        conn.execute("UPDATE documents SET created_at = '2026-11-23T01:00:00Z' WHERE id = ?;", (doc1.id,))
        conn.execute("UPDATE documents SET created_at = '2026-11-23T02:00:00Z' WHERE id = ?;", (doc2.id,))

    ordered_docs = service.list_documents()
    assert ordered_docs[0].id == doc2.id
    assert ordered_docs[1].id == doc1.id

    fetched = service.get_document(doc1.id)
    assert fetched.id == doc1.id
    assert fetched.filename == "first.md"
