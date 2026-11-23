"""Domain models for extracted pages, documents, and deterministic chunks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    """Extracted text unit with optional 1-based physical page number.

    Attributes:
        text: Non-empty extracted text string.
        page_number: 1-indexed integer for PDF pages; None for Markdown/TXT.
    """

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    """Extracted document containing ordered extracted pages.

    Attributes:
        filename: Safe filename of the document.
        file_type: Normalized file type ('pdf', 'md', 'txt').
        content_sha256: Hexadecimal SHA-256 digest of original bytes.
        pages: Ordered list of extracted non-empty pages.
    """

    filename: str
    file_type: str
    content_sha256: str
    pages: list[ExtractedPage]

    @property
    def total_characters(self) -> int:
        """Total character count across all pages."""
        return sum(len(p.text) for p in self.pages)


@dataclass(frozen=True)
class DocumentChunk:
    """Deterministic character chunk representation.

    Attributes:
        chunk_index: Sequential 0-based integer across the entire document.
        text: Chunk text content.
        page_number: 1-indexed page number for PDF chunks; None for Markdown/TXT.
        document_id: Optional UUID string identifying the parent document.
    """

    chunk_index: int
    text: str
    page_number: int | None = None
    document_id: str | None = None

    @property
    def chunk_id(self) -> str | None:
        """Return formatted chunk ID (<document_uuid>:<chunk_index>) if document_id is set."""
        if self.document_id is not None:
            return f"{self.document_id}:{self.chunk_index}"
        return None
