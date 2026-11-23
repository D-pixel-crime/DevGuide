"""Comprehensive unit and integration tests for DocumentRepository."""

import pathlib
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DuplicateDocumentError,
)
from app.db.sqlite import get_connection, init_db
from app.models.document import Document


@pytest.fixture
def temp_db_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide an initialized temporary SQLite database path."""
    db_file = tmp_path / "test_devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)
    return db_file


@pytest.fixture
def repository(temp_db_path: pathlib.Path) -> DocumentRepository:
    """Provide a DocumentRepository bound to a temporary SQLite database."""
    return DocumentRepository(db_path=temp_db_path)


# ============================================================================
# CREATE TESTS
# ============================================================================


def test_create_document_success(repository: DocumentRepository) -> None:
    """Verify successful document creation with correct attributes."""
    doc = repository.create_document(
        filename="overview.md",
        file_type="md",
        content_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        chunk_count=3,
    )

    assert isinstance(doc, Document)
    assert doc.filename == "overview.md"
    assert doc.file_type == "md"
    assert (
        doc.content_sha256
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert doc.chunk_count == 3
    assert isinstance(doc.id, str)
    assert isinstance(doc.created_at, str)


def test_create_document_generated_uuid_validity(repository: DocumentRepository) -> None:
    """Verify generated UUID is a valid RFC 4122 v4 UUID string."""
    doc = repository.create_document(
        filename="spec.pdf",
        file_type="pdf",
        content_sha256="hash_uuid_test_1",
        chunk_count=10,
    )

    parsed = uuid.UUID(doc.id)
    assert str(parsed) == doc.id
    assert parsed.version == 4


def test_create_document_supplied_uuid_validity(repository: DocumentRepository) -> None:
    """Verify supplied valid UUID (string or uuid.UUID) is preserved, and invalid UUIDs are rejected."""
    # Supplied valid string UUID
    supplied_id_str = str(uuid.uuid4())
    doc1 = repository.create_document(
        filename="architecture.txt",
        file_type="txt",
        content_sha256="hash_uuid_test_2",
        chunk_count=2,
        document_id=supplied_id_str,
    )
    assert doc1.id == supplied_id_str

    # Supplied valid uuid.UUID object
    supplied_id_obj = uuid.uuid4()
    doc2 = repository.create_document(
        filename="design.md",
        file_type="md",
        content_sha256="hash_uuid_test_3",
        chunk_count=4,
        document_id=supplied_id_obj,
    )
    assert doc2.id == str(supplied_id_obj)

    # Invalid string UUID rejected
    with pytest.raises(ValueError, match="Invalid UUID"):
        repository.create_document(
            filename="invalid.txt",
            file_type="txt",
            content_sha256="hash_invalid_uuid",
            chunk_count=1,
            document_id="not-a-valid-uuid",
        )

    # Empty string UUID rejected
    with pytest.raises(ValueError, match="Document ID must be a non-empty string or UUID"):
        repository.create_document(
            filename="invalid_empty.txt",
            file_type="txt",
            content_sha256="hash_invalid_uuid_empty",
            chunk_count=1,
            document_id="   ",
        )

    # Non-string/non-UUID rejected
    with pytest.raises(ValueError, match="Document ID must be a non-empty string or UUID"):
        repository.create_document(
            filename="invalid2.txt",
            file_type="txt",
            content_sha256="hash_invalid_uuid_2",
            chunk_count=1,
            document_id=12345,  # type: ignore[arg-type]
        )


def test_create_document_utc_timestamp_format(repository: DocumentRepository) -> None:
    """Verify generated created_at is UTC ISO-8601 formatted with Z suffix."""
    doc = repository.create_document(
        filename="report.pdf",
        file_type="pdf",
        content_sha256="hash_utc_test",
        chunk_count=5,
    )

    assert doc.created_at.endswith("Z")
    parsed_dt = datetime.fromisoformat(doc.created_at)
    assert parsed_dt.tzinfo == timezone.utc


def test_create_document_negative_chunk_count_rejection(
    repository: DocumentRepository,
) -> None:
    """Verify negative chunk_count is rejected with ValueError before SQL execution."""
    with pytest.raises(ValueError, match="chunk_count must be non-negative"):
        repository.create_document(
            filename="bad_chunks.md",
            file_type="md",
            content_sha256="hash_neg_chunks_1",
            chunk_count=-1,
        )

    with pytest.raises(ValueError, match="chunk_count must be non-negative"):
        repository.create_document(
            filename="bad_chunks.md",
            file_type="md",
            content_sha256="hash_neg_chunks_2",
            chunk_count=-999,
        )


def test_create_document_duplicate_content_sha256_rejection(
    repository: DocumentRepository,
) -> None:
    """Verify duplicate content_sha256 raises DuplicateDocumentError preserving original cause."""
    hash_val = "duplicate_hash_alpha"
    repository.create_document(
        filename="first.md",
        file_type="md",
        content_sha256=hash_val,
        chunk_count=1,
    )

    with pytest.raises(DuplicateDocumentError, match="already exists") as exc_info:
        repository.create_document(
            filename="second.md",
            file_type="md",
            content_sha256=hash_val,
            chunk_count=2,
        )

    # Verify cause is preserved
    assert isinstance(exc_info.value.__cause__, sqlite3.IntegrityError)

    # Verify only the first document is present
    docs = repository.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "first.md"


def test_create_document_unrelated_constraint_not_misclassified(
    repository: DocumentRepository,
) -> None:
    """Verify unrelated constraint violations (e.g. duplicate primary key) raise IntegrityError, not DuplicateDocumentError."""
    fixed_id = str(uuid.uuid4())
    repository.create_document(
        filename="doc1.md",
        file_type="md",
        content_sha256="hash_pk_1",
        chunk_count=1,
        document_id=fixed_id,
    )

    # Attempting to insert same ID with DIFFERENT content_sha256
    with pytest.raises(sqlite3.IntegrityError) as exc_info:
        repository.create_document(
            filename="doc2.md",
            file_type="md",
            content_sha256="hash_pk_2",
            chunk_count=2,
            document_id=fixed_id,
        )

    # Must NOT be DuplicateDocumentError
    assert not isinstance(exc_info.value, DuplicateDocumentError)
    assert "id" in str(exc_info.value).lower() or "primary" in str(exc_info.value).lower()


# ============================================================================
# LIST TESTS
# ============================================================================


def test_list_documents_empty(repository: DocumentRepository) -> None:
    """Verify list_documents returns empty list on empty database."""
    docs = repository.list_documents()
    assert docs == []


def test_list_documents_multiple(repository: DocumentRepository) -> None:
    """Verify list_documents retrieves all stored documents."""
    for i in range(3):
        repository.create_document(
            filename=f"doc_{i}.md",
            file_type="md",
            content_sha256=f"hash_list_{i}",
            chunk_count=i + 1,
        )

    docs = repository.list_documents()
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)


def test_list_documents_newest_first_ordering(
    repository: DocumentRepository, temp_db_path: pathlib.Path
) -> None:
    """Verify documents are returned ordered newest-first by created_at using assigned timestamps."""
    doc1 = repository.create_document(
        filename="oldest.md",
        file_type="md",
        content_sha256="hash_oldest",
        chunk_count=1,
    )
    doc2 = repository.create_document(
        filename="newest.md",
        file_type="md",
        content_sha256="hash_newest",
        chunk_count=1,
    )
    doc3 = repository.create_document(
        filename="middle.md",
        file_type="md",
        content_sha256="hash_middle",
        chunk_count=1,
    )

    # Explicit SQL updates through temporary connection to assign known timestamps
    with get_connection(temp_db_path) as conn:
        conn.execute(
            "UPDATE documents SET created_at = ? WHERE id = ?;",
            ("2026-11-23T01:00:00Z", doc1.id),
        )
        conn.execute(
            "UPDATE documents SET created_at = ? WHERE id = ?;",
            ("2026-11-23T03:00:00Z", doc2.id),
        )
        conn.execute(
            "UPDATE documents SET created_at = ? WHERE id = ?;",
            ("2026-11-23T02:00:00Z", doc3.id),
        )

    docs = repository.list_documents()
    assert [d.filename for d in docs] == ["newest.md", "middle.md", "oldest.md"]


def test_list_documents_deterministic_secondary_ordering(
    repository: DocumentRepository, temp_db_path: pathlib.Path
) -> None:
    """Verify equal timestamps use deterministic secondary ordering by id ASC."""
    shared_timestamp = "2026-11-23T12:00:00Z"
    id_a = "00000000-0000-0000-0000-000000000001"
    id_b = "00000000-0000-0000-0000-000000000002"

    # Insert id_b first, then id_a
    repository.create_document(
        filename="second_id.md",
        file_type="md",
        content_sha256="hash_equal_b",
        chunk_count=1,
        document_id=id_b,
    )
    repository.create_document(
        filename="first_id.md",
        file_type="md",
        content_sha256="hash_equal_a",
        chunk_count=1,
        document_id=id_a,
    )

    # Assign identical created_at to test deterministic secondary ordering
    with get_connection(temp_db_path) as conn:
        conn.execute("UPDATE documents SET created_at = ?;", (shared_timestamp,))

    docs = repository.list_documents()
    assert len(docs) == 2
    assert docs[0].id == id_a
    assert docs[1].id == id_b


# ============================================================================
# GET TESTS
# ============================================================================


def test_get_document_existing(repository: DocumentRepository) -> None:
    """Verify retrieving existing document by UUID (string and uuid.UUID) returns typed Document."""
    created = repository.create_document(
        filename="find_me.pdf",
        file_type="pdf",
        content_sha256="hash_find_me",
        chunk_count=7,
    )

    # Retrieve by string ID
    fetched_by_str = repository.get_document(created.id)
    assert isinstance(fetched_by_str, Document)
    assert fetched_by_str == created

    # Retrieve by uuid.UUID object
    fetched_by_obj = repository.get_document(uuid.UUID(created.id))
    assert isinstance(fetched_by_obj, Document)
    assert fetched_by_obj == created


def test_get_document_missing(repository: DocumentRepository) -> None:
    """Verify get_document raises DocumentNotFoundError for absent document."""
    random_uuid = str(uuid.uuid4())
    with pytest.raises(DocumentNotFoundError, match=random_uuid):
        repository.get_document(random_uuid)


def test_get_document_invalid_uuid(repository: DocumentRepository) -> None:
    """Verify get_document validates UUID and raises ValueError for invalid UUID format."""
    with pytest.raises(ValueError, match="Invalid UUID"):
        repository.get_document("not-a-uuid")

    with pytest.raises(ValueError, match="Document ID must be a non-empty string or UUID"):
        repository.get_document(None)  # type: ignore[arg-type]


# ============================================================================
# DUPLICATE DETECTION TESTS
# ============================================================================


def test_duplicate_detection_existing_content_hash(
    repository: DocumentRepository,
) -> None:
    """Verify duplicate detection methods for existing content hash."""
    target_hash = "existing_content_sha_123"
    created = repository.create_document(
        filename="existing.txt",
        file_type="txt",
        content_sha256=target_hash,
        chunk_count=5,
    )

    assert repository.exists_by_content_sha256(target_hash) is True
    fetched = repository.get_document_by_content_sha256(target_hash)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.content_sha256 == target_hash


def test_duplicate_detection_non_existing_content_hash(
    repository: DocumentRepository,
) -> None:
    """Verify duplicate detection methods for non-existing content hash."""
    absent_hash = "absent_content_sha_999"
    assert repository.exists_by_content_sha256(absent_hash) is False
    assert repository.get_document_by_content_sha256(absent_hash) is None


def test_duplicate_detection_consistency(repository: DocumentRepository) -> None:
    """Verify exists_by_content_sha256 and get_document_by_content_sha256 consistency."""
    h1 = "hash_consistent_1"
    h2 = "hash_consistent_2"

    # Initially neither exists
    assert repository.exists_by_content_sha256(h1) is False
    assert repository.get_document_by_content_sha256(h1) is None
    assert repository.exists_by_content_sha256(h2) is False
    assert repository.get_document_by_content_sha256(h2) is None

    # Insert h1
    repository.create_document(filename="h1.txt", file_type="txt", content_sha256=h1, chunk_count=1)

    assert repository.exists_by_content_sha256(h1) is True
    doc1 = repository.get_document_by_content_sha256(h1)
    assert doc1 is not None
    assert doc1.content_sha256 == h1

    # h2 still absent
    assert repository.exists_by_content_sha256(h2) is False
    assert repository.get_document_by_content_sha256(h2) is None


# ============================================================================
# DELETE TESTS
# ============================================================================


def test_delete_document_success(repository: DocumentRepository) -> None:
    """Verify successful deletion returns True for string and uuid.UUID IDs."""
    doc = repository.create_document(
        filename="delete_target.pdf",
        file_type="pdf",
        content_sha256="hash_to_delete",
        chunk_count=3,
    )

    result = repository.delete_document(doc.id)
    assert result is True

    # Test deletion using uuid.UUID object
    doc2 = repository.create_document(
        filename="delete_target2.pdf",
        file_type="pdf",
        content_sha256="hash_to_delete_2",
        chunk_count=1,
    )
    result2 = repository.delete_document(uuid.UUID(doc2.id))
    assert result2 is True


def test_delete_document_missing(repository: DocumentRepository) -> None:
    """Verify deleting a missing document raises DocumentNotFoundError."""
    random_uuid = str(uuid.uuid4())
    with pytest.raises(DocumentNotFoundError, match=random_uuid):
        repository.delete_document(random_uuid)


def test_delete_document_no_longer_returned(repository: DocumentRepository) -> None:
    """Verify document is no longer returned by any repository query after deletion."""
    target_hash = "hash_deleted_lifecycle"
    doc = repository.create_document(
        filename="lifecycle.md",
        file_type="md",
        content_sha256=target_hash,
        chunk_count=2,
    )

    assert repository.exists_by_content_sha256(target_hash) is True
    assert len(repository.list_documents()) == 1

    # Delete
    repository.delete_document(doc.id)

    # Ensure no longer accessible
    with pytest.raises(DocumentNotFoundError):
        repository.get_document(doc.id)

    assert repository.exists_by_content_sha256(target_hash) is False
    assert repository.get_document_by_content_sha256(target_hash) is None
    assert repository.list_documents() == []


def test_delete_document_invalid_uuid(repository: DocumentRepository) -> None:
    """Verify delete_document validates UUID and raises ValueError for invalid UUID format."""
    with pytest.raises(ValueError, match="Invalid UUID"):
        repository.delete_document("invalid-uuid-string")


# ============================================================================
# PERSISTENCE TESTS
# ============================================================================


def test_persistence_across_separate_instances_and_connections(
    temp_db_path: pathlib.Path,
) -> None:
    """Verify data remains available across separate repository instances and connection scopes."""
    repo1 = DocumentRepository(db_path=temp_db_path)
    created_doc = repo1.create_document(
        filename="persistent_doc.md",
        file_type="md",
        content_sha256="hash_persistence_1",
        chunk_count=8,
    )

    # Instantiate brand new repository instance pointing to same database
    repo2 = DocumentRepository(db_path=temp_db_path)

    # Verify retrieval
    fetched = repo2.get_document(created_doc.id)
    assert fetched == created_doc

    # Verify listing
    all_docs = repo2.list_documents()
    assert len(all_docs) == 1
    assert all_docs[0] == created_doc

    # Verify duplicate detection
    assert repo2.exists_by_content_sha256("hash_persistence_1") is True
    assert repo2.get_document_by_content_sha256("hash_persistence_1") == created_doc
