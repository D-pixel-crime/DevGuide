"""SQLite document metadata repository for DevGuide."""

import pathlib
import sqlite3
import uuid
from datetime import datetime, timezone

from app.db.sqlite import get_connection
from app.models.document import Document


class DocumentRepositoryError(Exception):
    """Base exception for document repository errors."""


class DocumentNotFoundError(DocumentRepositoryError):
    """Raised when a requested document does not exist."""


class DuplicateDocumentError(DocumentRepositoryError):
    """Raised when a document with duplicate content_sha256 is inserted."""


def _validate_uuid(val: str | uuid.UUID) -> str:
    """Validate and return normalized canonical UUID string.

    Raises:
        ValueError: If val is not a valid UUID string or uuid.UUID object.
    """
    if isinstance(val, uuid.UUID):
        return str(val)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Document ID must be a non-empty string or UUID, got {val!r}")
    try:
        parsed = uuid.UUID(val.strip())
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid UUID: {val!r}") from exc
    return str(parsed)


class DocumentRepository:
    """Repository for managing document metadata in SQLite."""

    def __init__(self, db_path: pathlib.Path | str | None = None) -> None:
        self.db_path = pathlib.Path(db_path) if db_path is not None else None

    def create_document(
        self,
        filename: str,
        file_type: str,
        content_sha256: str,
        chunk_count: int,
        document_id: str | uuid.UUID | None = None,
    ) -> Document:
        """Create and store a new document record in SQLite.

        Args:
            filename: Document filename.
            file_type: Document file extension/type ('pdf', 'md', 'txt').
            content_sha256: SHA-256 hash of document raw content.
            chunk_count: Non-negative number of chunks.
            document_id: Optional UUID for the document (generated if omitted).

        Returns:
            Created typed Document instance.

        Raises:
            ValueError: If chunk_count < 0 or document_id is invalid.
            DuplicateDocumentError: If content_sha256 already exists.
            sqlite3.IntegrityError: For non-duplicate constraint violations.
        """
        if chunk_count < 0:
            raise ValueError(f"chunk_count must be non-negative, got {chunk_count}")

        final_id = str(uuid.uuid4()) if document_id is None else _validate_uuid(document_id)
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        insert_sql = """
        INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """.strip()

        try:
            with get_connection(self.db_path) as conn:
                conn.execute(
                    insert_sql,
                    (
                        final_id,
                        filename,
                        file_type,
                        content_sha256,
                        chunk_count,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            err_msg = str(exc)
            err_name = getattr(exc, "sqlite_errorname", "")
            err_code = getattr(exc, "sqlite_errorcode", None)
            is_unique_constraint = (
                err_name == "SQLITE_CONSTRAINT_UNIQUE"
                or err_code == getattr(sqlite3, "SQLITE_CONSTRAINT_UNIQUE", 2067)
                or "UNIQUE constraint failed:" in err_msg
            )
            is_content_sha_target = (
                "documents.content_sha256" in err_msg
                or err_msg.endswith(": content_sha256")
            )
            if is_unique_constraint and is_content_sha_target and "documents.id" not in err_msg:
                raise DuplicateDocumentError(
                    f"Document with content_sha256 '{content_sha256}' already exists"
                ) from exc
            raise

        return Document(
            id=final_id,
            filename=filename,
            file_type=file_type,
            content_sha256=content_sha256,
            chunk_count=chunk_count,
            created_at=created_at,
        )

    def list_documents(self) -> list[Document]:
        """List all documents newest-first with deterministic secondary ordering."""
        sql = """
        SELECT id, filename, file_type, content_sha256, chunk_count, created_at
        FROM documents
        ORDER BY created_at DESC, id ASC;
        """.strip()
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            return [
                Document(
                    id=row["id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    content_sha256=row["content_sha256"],
                    chunk_count=row["chunk_count"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def get_document(self, document_id: str | uuid.UUID) -> Document:
        """Retrieve a document by UUID."""
        doc_id = _validate_uuid(document_id)
        sql = """
        SELECT id, filename, file_type, content_sha256, chunk_count, created_at
        FROM documents
        WHERE id = ?;
        """.strip()
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql, (doc_id,))
            row = cursor.fetchone()
            if row is None:
                raise DocumentNotFoundError(f"Document with id '{doc_id}' not found")
            return Document(
                id=row["id"],
                filename=row["filename"],
                file_type=row["file_type"],
                content_sha256=row["content_sha256"],
                chunk_count=row["chunk_count"],
                created_at=row["created_at"],
            )

    def get_document_by_content_sha256(self, content_sha256: str) -> Document | None:
        """Retrieve a document by its content SHA-256 hash if present, or None."""
        sql = """
        SELECT id, filename, file_type, content_sha256, chunk_count, created_at
        FROM documents
        WHERE content_sha256 = ?;
        """.strip()
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql, (content_sha256,))
            row = cursor.fetchone()
            if row is None:
                return None
            return Document(
                id=row["id"],
                filename=row["filename"],
                file_type=row["file_type"],
                content_sha256=row["content_sha256"],
                chunk_count=row["chunk_count"],
                created_at=row["created_at"],
            )

    def exists_by_content_sha256(self, content_sha256: str) -> bool:
        """Check if a document exists by its content SHA-256 without fetching columns."""
        sql = "SELECT 1 FROM documents WHERE content_sha256 = ? LIMIT 1;"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql, (content_sha256,))
            row = cursor.fetchone()
            return row is not None

    def delete_document(self, document_id: str | uuid.UUID) -> bool:
        """Delete a document by UUID. Returns True on success, raises DocumentNotFoundError if missing."""
        doc_id = _validate_uuid(document_id)
        sql = "DELETE FROM documents WHERE id = ?;"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql, (doc_id,))
            if cursor.rowcount == 0:
                raise DocumentNotFoundError(f"Document with id '{doc_id}' not found")
            return True
