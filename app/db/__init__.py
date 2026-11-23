"""Database and storage adapters for DevGuide."""

from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DocumentRepositoryError,
    DuplicateDocumentError,
)
from app.db.sqlite import (
    DOCUMENTS_TABLE_NAME,
    check_sqlite_readiness,
    get_connection,
    init_db,
)

__all__ = [
    "DOCUMENTS_TABLE_NAME",
    "DocumentNotFoundError",
    "DocumentRepository",
    "DocumentRepositoryError",
    "DuplicateDocumentError",
    "check_sqlite_readiness",
    "get_connection",
    "init_db",
]
