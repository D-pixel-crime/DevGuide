"""Validated upload domain model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatedUpload:
    """Validated in-memory document upload representation.

    Attributes:
        filename: Safe, sanitized filename with supported extension.
        file_type: Normalized lowercase extension ('pdf', 'md', 'txt').
        content: Exact raw original bytes of the uploaded file.
        content_sha256: Hexadecimal SHA-256 digest computed from content.
    """

    filename: str
    file_type: str
    content: bytes
    content_sha256: str
