"""Document metadata domain model."""

from dataclasses import dataclass


@dataclass
class Document:
    """Document metadata model aligned exactly with SQLite documents table."""

    id: str
    filename: str
    file_type: str
    content_sha256: str
    chunk_count: int
    created_at: str
