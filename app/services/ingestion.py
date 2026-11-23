"""Document upload validation, text extraction, character chunking, and ingestion orchestration."""

import hashlib
import pathlib
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import pymupdf

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.chroma import ChromaAdapter, ChunkRecord, ChromaStorageError
from app.db.repository import (
    DocumentNotFoundError,
    DocumentRepository,
    DuplicateDocumentError,
)
from app.models.chunk import DocumentChunk, ExtractedDocument, ExtractedPage
from app.models.document import Document
from app.models.upload import ValidatedUpload
from app.services.embedding import OpenAIEmbeddingAdapter

logger = get_logger("app.services.ingestion")

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


class IngestionStorageError(IngestionError):
    """Raised when storage operations fail during document ingestion."""


class PartialConsistencyError(IngestionStorageError):
    """Raised when Chroma vectors persist but SQLite registration and cleanup fails."""


class ChromaDeletionError(IngestionError):
    """Raised when ChromaDB vector deletion fails during document deletion."""


class DeletionPartialConsistencyError(IngestionError):
    """Raised when Chroma vectors were deleted but SQLite metadata deletion failed."""


@dataclass(frozen=True)
class DocumentDeletionResult:
    """Result of a document deletion operation across ChromaDB and SQLite."""

    id: str
    deleted: bool
    chunks_removed: int


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


class IngestionService:
    """Orchestration service for document ingestion, retrieval, and deletion.

    Coordinates document upload validation, duplicate short-circuiting, text extraction,
    deterministic chunking, vector embedding generation, ChromaDB vector indexing,
    and SQLite metadata registration with partial failure compensation.
    """

    def __init__(
        self,
        repository: Optional[DocumentRepository] = None,
        chroma_adapter: Optional[ChromaAdapter] = None,
        embedding_adapter: Optional[OpenAIEmbeddingAdapter] = None,
        settings: Optional[Settings] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """Initialize IngestionService with optional injectable dependencies."""
        self._settings = settings if settings is not None else get_settings()
        self._repository = (
            repository
            if repository is not None
            else DocumentRepository(db_path=self._settings.SQLITE_PATH)
        )
        self._chroma_adapter = (
            chroma_adapter
            if chroma_adapter is not None
            else ChromaAdapter(
                chroma_path=self._settings.CHROMA_PATH,
                settings=self._settings,
            )
        )
        self._embedding_adapter = (
            embedding_adapter
            if embedding_adapter is not None
            else OpenAIEmbeddingAdapter(settings=self._settings)
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def repository(self) -> DocumentRepository:
        """Configured SQLite document repository."""
        return self._repository

    @property
    def chroma_adapter(self) -> ChromaAdapter:
        """Configured ChromaDB vector-store adapter."""
        return self._chroma_adapter

    @property
    def embedding_adapter(self) -> OpenAIEmbeddingAdapter:
        """Configured OpenAI embedding adapter."""
        return self._embedding_adapter

    @property
    def chunk_size(self) -> int:
        """Resolved character chunk size."""
        if self._chunk_size is not None:
            return self._chunk_size
        return self._settings.CHUNK_SIZE

    @property
    def chunk_overlap(self) -> int:
        """Resolved character chunk overlap."""
        if self._chunk_overlap is not None:
            return self._chunk_overlap
        return self._settings.CHUNK_OVERLAP

    def ingest_document(
        self,
        filename: str,
        content: bytes | bytearray,
    ) -> Document:
        """Orchestrate end-to-end document ingestion.

        Pipeline:
        1. validate_upload: validates filename, size, file_type, content consistency, sha256.
        2. Duplicate check by SHA-256 against SQLite; short-circuit before expensive embedding work.
        3. extract_text: extracts text from PDF (PyMuPDF) or MD/TXT (UTF-8).
        4. Generate document UUID.
        5. chunk_document with document_id, preserving page boundaries.
        6. Embed chunks using OpenAI embedding adapter.
        7. Insert chunk records into ChromaDB vector store.
        8. Register document metadata in SQLite documents table.
           If Chroma vector insertion fails, SQLite metadata is never registered.
           If Chroma succeeds but SQLite registration fails, attempt best-effort vector deletion.
           If vector cleanup fails, report PartialConsistencyError.
        9. Store document-level metadata only; never persist original upload bytes.

        Args:
            filename: Original uploaded filename.
            content: Raw uploaded file bytes.

        Returns:
            Created typed Document instance.

        Raises:
            FileValidationError: If file validation checks fail.
            DuplicateDocumentError: If document with identical SHA-256 already exists.
            TextExtractionError: If text extraction fails.
            ChunkingError: If chunking fails or produces zero chunks.
            EmbeddingError: If embedding generation fails.
            ChromaStorageError: If ChromaDB vector write fails.
            PartialConsistencyError: If Chroma succeeds but SQLite fails and cleanup fails.
            IngestionStorageError: If Chroma succeeds but SQLite registration fails and cleanup succeeds.
        """
        # 1. Validation
        upload = validate_upload(
            filename=filename,
            content=content,
            max_upload_mb=self._settings.MAX_UPLOAD_MB,
        )

        # 2. Duplicate check by SHA-256 before expensive extraction and embedding
        existing_doc = self._repository.get_document_by_content_sha256(upload.content_sha256)
        if existing_doc is not None:
            raise DuplicateDocumentError(
                f"Document with identical content SHA-256 hash already exists (ID: {existing_doc.id})"
            )

        # 3. Extraction
        extracted_doc = extract_text(upload)

        # 4. Generate document UUID
        doc_id = str(uuid.uuid4())

        # 5. Chunking
        chunks = chunk_document(
            extracted_doc,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            document_id=doc_id,
        )
        if not chunks:
            raise EmptyExtractionError("Document produced 0 text chunks")

        # 6. Embedding
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_adapter.embed_texts(texts)

        # 7. Insert Chroma vectors
        chunk_records = [
            ChunkRecord.from_document_chunk(
                chunk=chunk,
                embedding=emb,
                filename=upload.filename,
                file_type=upload.file_type,
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self._chroma_adapter.add_chunks(chunk_records)

        # 8. Register SQLite metadata with compensation on failure
        try:
            doc = self._repository.create_document(
                filename=upload.filename,
                file_type=upload.file_type,
                content_sha256=upload.content_sha256,
                chunk_count=len(chunks),
                document_id=doc_id,
            )
        except Exception as sql_exc:
            logger.error(
                "SQLite metadata insertion failed for document %s: %s. Initiating best-effort cleanup.",
                doc_id,
                type(sql_exc).__name__,
            )
            cleanup_failed = False
            try:
                self._chroma_adapter.delete_records_by_document(doc_id)
                logger.info("Best-effort vector cleanup succeeded for document %s", doc_id)
            except Exception as cleanup_exc:
                cleanup_failed = True
                logger.critical(
                    "Best-effort vector cleanup failed for document %s: %s",
                    doc_id,
                    type(cleanup_exc).__name__,
                )

            if cleanup_failed:
                raise PartialConsistencyError(
                    "Partial consistency failure: vectors persisted in ChromaDB but SQLite metadata insertion failed and vector cleanup could not be completed"
                ) from sql_exc

            if isinstance(sql_exc, DuplicateDocumentError):
                raise sql_exc

            raise IngestionStorageError(
                f"Failed to register document metadata in SQLite: {type(sql_exc).__name__}"
            ) from sql_exc

        return doc

    def delete_document(self, document_id: str) -> DocumentDeletionResult:
        """Delete a document and its corresponding vector chunks.

        1. Verify existence in SQLite. Raises DocumentNotFoundError (404) if missing.
        2. Delete Chroma vectors first using canonical doc.id. If Chroma deletion fails, abort and preserve SQLite metadata.
        3. Delete SQLite metadata using canonical doc.id. If Chroma succeeds but SQLite fails, report PartialConsistencyError.
        4. Return DocumentDeletionResult with canonical doc.id.

        Args:
            document_id: UUID string of the document to delete.

        Returns:
            DocumentDeletionResult with canonical lowercase id, deleted=True, and chunks_removed.

        Raises:
            DocumentNotFoundError: If document does not exist in SQLite.
            ValueError: If document_id is not a valid UUID string.
            ChromaDeletionError: If deleting vectors from ChromaDB fails.
            DeletionPartialConsistencyError: If Chroma vectors deleted but SQLite deletion fails.
        """
        # 1. Existence check in SQLite (normalizes document_id into canonical lowercase UUID)
        doc = self._repository.get_document(document_id)
        canonical_id = doc.id

        # 2. Delete Chroma vectors first using canonical doc.id
        try:
            chunks_removed = self._chroma_adapter.delete_records_by_document(canonical_id)
        except Exception as chroma_exc:
            logger.error(
                "Failed to delete ChromaDB vectors for document %s: %s. Aborting metadata deletion.",
                canonical_id,
                type(chroma_exc).__name__,
            )
            raise ChromaDeletionError(
                "Failed to delete vectors from ChromaDB. Document metadata deletion aborted."
            ) from chroma_exc

        # 3. Delete SQLite metadata using canonical doc.id
        try:
            self._repository.delete_document(canonical_id)
        except Exception as sql_exc:
            logger.critical(
                "Partial consistency failure: vectors deleted from ChromaDB for document %s, but failed to delete SQLite metadata record: %s",
                canonical_id,
                type(sql_exc).__name__,
            )
            raise DeletionPartialConsistencyError(
                "Partial consistency failure: vectors deleted from ChromaDB but failed to delete SQLite metadata record."
            ) from sql_exc

        return DocumentDeletionResult(
            id=canonical_id,
            deleted=True,
            chunks_removed=chunks_removed,
        )

    def get_document(self, document_id: str) -> Document:
        """Retrieve document metadata by document ID.

        Args:
            document_id: UUID string of the document.

        Returns:
            Document metadata model.

        Raises:
            DocumentNotFoundError: If document does not exist.
            ValueError: If document_id is not a valid UUID string.
        """
        return self._repository.get_document(document_id)

    def list_documents(self) -> list[Document]:
        """List all indexed documents ordered newest first.

        Returns:
            List of Document metadata models.
        """
        return self._repository.list_documents()
