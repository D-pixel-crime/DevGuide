"""Document upload validation, text extraction, and character chunking service."""

import hashlib
import pathlib
import unicodedata
from collections.abc import Sequence
from typing import Optional

import pymupdf

from app.core.config import get_settings
from app.models.chunk import DocumentChunk, ExtractedDocument, ExtractedPage
from app.models.upload import ValidatedUpload

# Supported file extensions
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".txt"})

# Windows reserved device names
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


class IngestionError(Exception):
    """Base exception for document ingestion, extraction, and chunking errors."""


class FileValidationError(IngestionError, ValueError):
    """Raised when an uploaded file fails validation checks."""


class UnsupportedFileTypeError(FileValidationError):
    """Raised when an uploaded file extension is not supported."""


class FileSizeLimitExceededError(FileValidationError):
    """Raised when an uploaded file exceeds the MAX_UPLOAD_MB size limit."""


class EmptyFileError(FileValidationError):
    """Raised when an uploaded file is empty (0 bytes)."""


class InvalidFilenameError(FileValidationError):
    """Raised when an uploaded file has an invalid or unsafe filename."""


class FileContentMismatchError(FileValidationError):
    """Raised when file content does not match its declared extension."""


class InvalidEncodingError(FileValidationError):
    """Raised when a text file is not strictly valid UTF-8."""


class TextExtractionError(IngestionError):
    """Raised when text extraction from a file fails."""


class CorruptPDFError(TextExtractionError):
    """Raised when a PDF file is corrupt or unreadable."""


class ImageOnlyPDFError(TextExtractionError):
    """Raised when a PDF contains only images and no extractable text."""


class EmptyExtractionError(TextExtractionError):
    """Raised when a file contains no extractable non-whitespace text."""


class ChunkingError(IngestionError, ValueError):
    """Raised when chunking parameters are invalid or chunking fails."""


def validate_filename(filename: str) -> str:
    """Validate and sanitize an uploaded document filename.

    Enforces filename safety: rejects empty/whitespace names, Unicode control characters
    (category Cc), colons (blocking drive-relative paths and NTFS alternate data streams),
    path traversal, path separators, hidden/dot files, Windows reserved device names,
    trailing dots/spaces, excessive length, and unsupported file extensions.

    Args:
        filename: Raw input filename.

    Returns:
        Safe, normalized filename string.

    Raises:
        InvalidFilenameError: If the filename is unsafe or malformed.
        UnsupportedFileTypeError: If the file extension is not supported.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise InvalidFilenameError("Filename must be a non-empty string")

    # Reject Unicode control characters (Unicode General Category 'Cc')
    if any(unicodedata.category(c) == "Cc" for c in filename):
        raise InvalidFilenameError("Filename contains invalid Unicode control characters")

    # Reject colon characters (blocks Windows drive-relative paths and NTFS alternate data streams)
    if ":" in filename:
        raise InvalidFilenameError(
            f"Filename must not contain colon characters: {filename!r}"
        )

    # Reject leading/trailing whitespace or trailing dots
    if filename.startswith(" ") or filename.endswith((" ", ".")):
        raise InvalidFilenameError(
            f"Filename must not have leading/trailing whitespace or trailing dots: {filename!r}"
        )

    # Reject path separators and path traversal
    if "/" in filename or "\\" in filename:
        raise InvalidFilenameError(
            f"Filename must not contain path separators or directory components: {filename!r}"
        )
    if ".." in filename:
        raise InvalidFilenameError(
            f"Filename must not contain path traversal sequences ('..'): {filename!r}"
        )

    clean_name = filename.strip()

    # Reject hidden files or files without stem (e.g. '.pdf', '.txt')
    if clean_name.startswith("."):
        raise InvalidFilenameError(
            f"Hidden files or extension-only filenames are not allowed: {filename!r}"
        )

    # Reject filenames exceeding standard filesystem length (255 chars)
    if len(clean_name) > 255:
        raise InvalidFilenameError(
            f"Filename exceeds maximum length of 255 characters (got {len(clean_name)})"
        )

    path = pathlib.PurePath(clean_name)
    stem_upper = path.stem.upper()
    if stem_upper in WINDOWS_RESERVED_NAMES:
        raise InvalidFilenameError(
            f"Filename stem {stem_upper!r} is a reserved system device name"
        )

    ext = path.suffix.lower()
    if not ext:
        raise UnsupportedFileTypeError(
            "Filename is missing a file extension. Allowed types: .pdf, .md, .txt"
        )

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Allowed types: .pdf, .md, .txt"
        )

    return clean_name


def validate_upload(
    filename: str,
    content: bytes | bytearray,
    max_upload_mb: Optional[int] = None,
) -> ValidatedUpload:
    """Validate raw file upload and construct an in-memory ValidatedUpload representation.

    Validates filename safety, extension, file size, content consistency, and encoding.
    Does not persist the file to disk. Computes SHA-256 digest from exact original bytes.

    Args:
        filename: Raw uploaded filename.
        content: Raw uploaded file bytes.
        max_upload_mb: Optional max upload size limit in MB (defaults to Settings.MAX_UPLOAD_MB).

    Returns:
        ValidatedUpload domain object.

    Raises:
        FileValidationError: For invalid filenames, sizes, content mismatches, or encodings.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise FileValidationError(
            f"File content must be bytes or bytearray, got {type(content).__name__}"
        )

    raw_bytes = bytes(content)
    safe_filename = validate_filename(filename)
    ext = pathlib.PurePath(safe_filename).suffix.lower()
    file_type = ext.lstrip(".")

    # Size validation
    settings = get_settings()
    limit_mb = max_upload_mb if max_upload_mb is not None else settings.MAX_UPLOAD_MB
    max_bytes = limit_mb * 1024 * 1024
    byte_len = len(raw_bytes)

    if byte_len == 0:
        raise EmptyFileError("File is empty (0 bytes)")

    if byte_len > max_bytes:
        raise FileSizeLimitExceededError(
            f"File size exceeds maximum allowed upload limit of {limit_mb} MB ({max_bytes} bytes)"
        )

    # Content consistency validation
    if file_type == "pdf":
        if b"%PDF-" not in raw_bytes[:1024]:
            raise FileContentMismatchError(
                "File extension is .pdf but content is missing valid %PDF- header"
            )
        try:
            with pymupdf.open(stream=raw_bytes, filetype="pdf") as doc:
                if not doc.is_pdf:
                    raise FileContentMismatchError(
                        "PyMuPDF failed to recognize stream as valid PDF"
                    )
        except (FileContentMismatchError, FileValidationError):
            raise
        except Exception as exc:
            raise CorruptPDFError(f"Malformed or corrupt PDF file: {exc}") from exc

    elif file_type in ("md", "txt"):
        if b"%PDF-" in raw_bytes[:1024]:
            raise FileContentMismatchError(
                f"File extension is .{file_type} but content contains PDF binary header"
            )
        if b"\x00" in raw_bytes:
            raise FileContentMismatchError(
                f"File extension is .{file_type} but content contains binary/null byte data"
            )
        try:
            raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise InvalidEncodingError(
                f"File content is not valid UTF-8: {exc}"
            ) from exc

    content_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    return ValidatedUpload(
        filename=safe_filename,
        file_type=file_type,
        content=raw_bytes,
        content_sha256=content_sha256,
    )


def extract_text(upload: ValidatedUpload) -> ExtractedDocument:
    """Extract text from a validated document upload.

    For PDF: Extracts text per page using PyMuPDF, preserving 1-based page numbers.
             Empty/unextractable or image-only PDFs raise explicit errors. No OCR.
    For MD/TXT: Decodes as UTF-8 string with page_number=None. Empty or whitespace-only
                content raises explicit error.

    Args:
        upload: ValidatedUpload domain object.

    Returns:
        ExtractedDocument containing ordered non-empty ExtractedPage objects.

    Raises:
        CorruptPDFError: If PDF structure is invalid, corrupt, or encrypted.
        ImageOnlyPDFError: If PDF contains images but zero extractable text.
        EmptyExtractionError: If extracted text is completely empty or whitespace-only.
        InvalidEncodingError: If text decoding fails.
        UnsupportedFileTypeError: If file_type is unknown.
    """
    if upload.file_type == "pdf":
        try:
            with pymupdf.open(stream=upload.content, filetype="pdf") as doc:
                if doc.is_encrypted:
                    raise CorruptPDFError("PDF document is encrypted or password-protected")

                if doc.page_count == 0:
                    raise EmptyExtractionError("PDF document has 0 pages")

                pages: list[ExtractedPage] = []
                has_images = False
                total_extracted_chars = 0

                for page_idx in range(doc.page_count):
                    page = doc[page_idx]
                    page_num = page_idx + 1  # 1-based physical page number

                    if not has_images and len(page.get_images()) > 0:
                        has_images = True

                    page_text = page.get_text()
                    if page_text and page_text.strip():
                        total_extracted_chars += len(page_text.strip())
                        pages.append(ExtractedPage(text=page_text, page_number=page_num))

                if total_extracted_chars == 0:
                    if has_images:
                        raise ImageOnlyPDFError(
                            "PDF contains only images and no extractable text; OCR is not supported"
                        )
                    raise EmptyExtractionError("PDF contains no extractable text")

                return ExtractedDocument(
                    filename=upload.filename,
                    file_type=upload.file_type,
                    content_sha256=upload.content_sha256,
                    pages=pages,
                )
        except (CorruptPDFError, ImageOnlyPDFError, EmptyExtractionError):
            raise
        except Exception as exc:
            raise CorruptPDFError(f"Failed to extract text from PDF: {exc}") from exc

    elif upload.file_type in ("md", "txt"):
        try:
            text = upload.content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise InvalidEncodingError(f"File content is not valid UTF-8: {exc}") from exc

        if not text.strip():
            raise EmptyExtractionError(
                "File contains no extractable text (empty or whitespace only)"
            )

        return ExtractedDocument(
            filename=upload.filename,
            file_type=upload.file_type,
            content_sha256=upload.content_sha256,
            pages=[ExtractedPage(text=text, page_number=None)],
        )

    raise UnsupportedFileTypeError(f"Unsupported file type: {upload.file_type!r}")


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Deterministically partition a text string into overlapping character chunks.

    Args:
        text: Text string to chunk.
        chunk_size: Character length of each chunk (defaults to Settings.CHUNK_SIZE).
        chunk_overlap: Overlap character count between consecutive chunks (defaults to Settings.CHUNK_OVERLAP).

    Returns:
        List of chunk strings. Returns empty list if text is empty.

    Raises:
        ChunkingError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    if size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {size}")
    if overlap < 0:
        raise ChunkingError(f"chunk_overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ChunkingError(
            f"chunk_overlap ({overlap}) must be strictly smaller than chunk_size ({size})"
        )

    if not text:
        return []

    text_len = len(text)
    if text_len <= size:
        return [text]

    chunks: list[str] = []
    step = size - overlap
    start = 0

    while start < text_len:
        end = min(start + size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += step

    return chunks


def chunk_document(
    document: ExtractedDocument | Sequence[ExtractedPage],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    document_id: Optional[str] = None,
) -> list[DocumentChunk]:
    """Chunk extracted document pages while preserving page boundaries and global chunk indexes.

    For PDF: Chunks each page independently without crossing page boundaries.
    For MD/TXT: Chunks document text with page_number=None.
    Maintains a sequential 0-based chunk_index across the entire document.

    Args:
        document: ExtractedDocument or sequence of ExtractedPage objects.
        chunk_size: Character length of each chunk (defaults to Settings.CHUNK_SIZE).
        chunk_overlap: Overlap character count between consecutive chunks (defaults to Settings.CHUNK_OVERLAP).
        document_id: Optional UUID of the document for formatting chunk IDs.

    Returns:
        List of DocumentChunk domain objects.

    Raises:
        ChunkingError: If chunking parameters are invalid.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    # Validate parameters early before processing pages
    if size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {size}")
    if overlap < 0:
        raise ChunkingError(f"chunk_overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ChunkingError(
            f"chunk_overlap ({overlap}) must be strictly smaller than chunk_size ({size})"
        )

    pages = document.pages if isinstance(document, ExtractedDocument) else list(document)
    all_chunks: list[DocumentChunk] = []
    global_chunk_index = 0

    for page in pages:
        text_slices = chunk_text(page.text, chunk_size=size, chunk_overlap=overlap)
        for text_slice in text_slices:
            all_chunks.append(
                DocumentChunk(
                    chunk_index=global_chunk_index,
                    text=text_slice,
                    page_number=page.page_number,
                    document_id=document_id,
                )
            )
            global_chunk_index += 1

    return all_chunks
