"""Comprehensive tests for upload validation, text extraction, and character chunking."""

import hashlib
import os
import pathlib
import tempfile
import uuid

import pymupdf
import pytest

from app.core.config import get_settings
from app.models.chunk import DocumentChunk, ExtractedDocument, ExtractedPage
from app.models.upload import ValidatedUpload
from app.services.ingestion import (
    ChunkingError,
    CorruptPDFError,
    EmptyExtractionError,
    EmptyFileError,
    FileContentMismatchError,
    FileSizeLimitExceededError,
    FileValidationError,
    ImageOnlyPDFError,
    IngestionError,
    InvalidEncodingError,
    InvalidFilenameError,
    UnsupportedFileTypeError,
    chunk_document,
    chunk_text,
    extract_text,
    validate_filename,
    validate_upload,
)


def _create_minimal_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate in-memory PDF bytes with given page texts."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_image_only_pdf() -> bytes:
    """Helper to generate an in-memory PDF containing an image but no text."""
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 50, 200, 200)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), 0)
    page.insert_image(rect, pixmap=pix)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ==============================================================================
# 1. Filename Safety and Path Traversal Tests
# ==============================================================================


@pytest.mark.parametrize(
    "valid_name, expected_ext",
    [
        ("document.pdf", ".pdf"),
        ("README.md", ".md"),
        ("notes.txt", ".txt"),
        ("my-report_2026.final.PDF", ".pdf"),
        ("quickstart.MD", ".md"),
        ("guide.TXT", ".txt"),
        ("document with spaces.pdf", ".pdf"),
    ],
)
def test_validate_filename_valid(valid_name: str, expected_ext: str):
    """Verify valid filenames normalize cleanly."""
    sanitized = validate_filename(valid_name)
    assert sanitized == valid_name
    assert sanitized.lower().endswith(expected_ext)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "",
        "   ",
        "\t",
        "\n",
        "../secret.pdf",
        "..\\secret.pdf",
        "nested/../../secret.txt",
        "/etc/passwd.txt",
        "C:\\Windows\\system32.md",
        "folder/file.pdf",
        "folder\\file.txt",
        "doc\x00.pdf",
        "doc\n.txt",
        "doc\r.md",
        ".pdf",
        ".md",
        ".txt",
        ".env",
        ".hidden.txt",
        "file.pdf.",
        "file.md ",
        "CON.txt",
        "prn.pdf",
        "AUX.md",
        "nul.txt",
        "com1.pdf",
        "lpt2.md",
        "C:secret.pdf",
        "report:stream.txt",
        "doc\x85.txt",
        "a" * 256 + ".pdf",
    ],
)
def test_validate_filename_unsafe_rejected(unsafe_name: str):
    """Verify unsafe filenames and path traversal attempts are rejected."""
    with pytest.raises(InvalidFilenameError):
        validate_filename(unsafe_name)


@pytest.mark.parametrize(
    "control_char_filename",
    [
        "doc\x01.pdf",
        "doc\x08.txt",
        "doc\x1b.md",
        "doc\x7f.pdf",
        "doc\x85.txt",
        "\x01file.pdf",
        "file\x85.md",
        "unsafe\x01",
        "unsafe\x08",
        "unsafe\x1b",
        "unsafe\x7f",
        "unsafe\x85",
    ],
)
def test_validate_filename_unicode_control_characters_rejected(control_char_filename: str):
    """Verify ASCII and C1 Unicode control characters (Cc) raise InvalidFilenameError."""
    with pytest.raises(InvalidFilenameError) as exc_info:
        validate_filename(control_char_filename)
    assert "control characters" in str(exc_info.value)


@pytest.mark.parametrize(
    "colon_filename",
    [
        "C:secret.pdf",
        "D:notes.txt",
        "report:stream.txt",
        "document.pdf:hidden",
        ":leading_colon.md",
        "C:secret",
        "report:stream",
    ],
)
def test_validate_filename_colon_syntax_rejected(colon_filename: str):
    """Verify colon syntax (drive-relative, NTFS alternate data streams) raises InvalidFilenameError."""
    with pytest.raises(InvalidFilenameError) as exc_info:
        validate_filename(colon_filename)
    assert "colon characters" in str(exc_info.value)


@pytest.mark.parametrize(
    "unsupported_name",
    [
        ("document.docx"),
        ("script.py"),
        ("data.json"),
        ("image.png"),
        ("archive.zip"),
        ("README"),
        ("no_extension"),
    ],
)
def test_validate_filename_unsupported_extensions(unsupported_name: str):
    """Verify unsupported extensions raise UnsupportedFileTypeError."""
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        validate_filename(unsupported_name)
    assert "Allowed types: .pdf, .md, .txt" in str(exc_info.value)


# ==============================================================================
# 2. Upload Validation Tests (Types, Sizes, Encodings, Content Consistency)
# ==============================================================================


def test_validate_upload_success_markdown():
    """Verify valid Markdown upload constructs ValidatedUpload correctly."""
    content = b"# Title\n\nSample markdown documentation."
    upload = validate_upload("guide.md", content)

    assert isinstance(upload, ValidatedUpload)
    assert upload.filename == "guide.md"
    assert upload.file_type == "md"
    assert upload.content == content
    assert upload.content_sha256 == hashlib.sha256(content).hexdigest()


def test_validate_upload_success_txt():
    """Verify valid TXT upload constructs ValidatedUpload correctly."""
    content = "Simple plain text file with Unicode: \u2705 \U0001f680".encode("utf-8")
    upload = validate_upload("notes.txt", content)

    assert upload.filename == "notes.txt"
    assert upload.file_type == "txt"
    assert upload.content == content
    assert upload.content_sha256 == hashlib.sha256(content).hexdigest()


def test_validate_upload_success_pdf():
    """Verify valid PDF upload constructs ValidatedUpload correctly."""
    pdf_bytes = _create_minimal_pdf(["Page 1 content"])
    upload = validate_upload("manual.pdf", pdf_bytes)

    assert upload.filename == "manual.pdf"
    assert upload.file_type == "pdf"
    assert upload.content == pdf_bytes
    assert upload.content_sha256 == hashlib.sha256(pdf_bytes).hexdigest()


def test_validate_upload_empty_file():
    """Verify empty 0-byte upload is rejected."""
    with pytest.raises(EmptyFileError) as exc_info:
        validate_upload("empty.txt", b"")
    assert "0 bytes" in str(exc_info.value)


def test_validate_upload_oversized_file():
    """Verify file exceeding MAX_UPLOAD_MB is rejected."""
    # 1 MB limit for test
    content = b"a" * (1 * 1024 * 1024 + 1)
    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        validate_upload("big.txt", content, max_upload_mb=1)
    assert "File size exceeds" in str(exc_info.value)


def test_validate_upload_exact_max_size_allowed():
    """Verify file exactly at MAX_UPLOAD_MB is accepted."""
    limit_mb = 1
    content = b"x" * (limit_mb * 1024 * 1024)
    upload = validate_upload("exact.txt", content, max_upload_mb=limit_mb)
    assert len(upload.content) == limit_mb * 1024 * 1024


def test_validate_upload_invalid_utf8_in_markdown():
    """Verify invalid UTF-8 bytes in Markdown are rejected without replacement."""
    # Invalid UTF-8 continuation byte
    bad_bytes = b"# Heading\n\x80\x81 Invalid UTF-8"
    with pytest.raises(InvalidEncodingError) as exc_info:
        validate_upload("broken.md", bad_bytes)
    assert "not valid UTF-8" in str(exc_info.value)


def test_validate_upload_invalid_utf8_in_txt():
    """Verify invalid UTF-8 in TXT is rejected."""
    bad_bytes = b"Hello \xff\xfe World"
    with pytest.raises(InvalidEncodingError) as exc_info:
        validate_upload("broken.txt", bad_bytes)
    assert "not valid UTF-8" in str(exc_info.value)


def test_validate_upload_pdf_disguised_as_txt():
    """Verify PDF binary data disguised with .txt extension is rejected."""
    pdf_bytes = _create_minimal_pdf(["Deceptive PDF"])
    with pytest.raises(FileContentMismatchError) as exc_info:
        validate_upload("fake.txt", pdf_bytes)
    assert "contains PDF binary header" in str(exc_info.value)


def test_validate_upload_pdf_disguised_as_md():
    """Verify PDF binary data disguised with .md extension is rejected."""
    pdf_bytes = _create_minimal_pdf(["Deceptive PDF"])
    with pytest.raises(FileContentMismatchError) as exc_info:
        validate_upload("fake.md", pdf_bytes)
    assert "contains PDF binary header" in str(exc_info.value)


def test_validate_upload_binary_null_bytes_in_text():
    """Verify binary files with null bytes disguised as text are rejected."""
    binary_content = b"Normal start\x00\x01\x02binary payload"
    with pytest.raises(FileContentMismatchError) as exc_info:
        validate_upload("fake.txt", binary_content)
    assert "contains binary/null byte" in str(exc_info.value)


def test_validate_upload_non_pdf_content_with_pdf_extension():
    """Verify text or random bytes with .pdf extension are rejected."""
    fake_pdf = b"This is just plain text, not a PDF."
    with pytest.raises(FileContentMismatchError) as exc_info:
        validate_upload("fake.pdf", fake_pdf)
    assert "missing valid %PDF- header" in str(exc_info.value)


def test_validate_upload_corrupt_pdf():
    """Verify corrupt PDF content is rejected."""
    corrupt_pdf = b"%PDF-1.4\nCorrupt body with invalid object table %%EOF"
    with pytest.raises(CorruptPDFError):
        validate_upload("corrupt.pdf", corrupt_pdf)


def test_validate_upload_content_type_non_bytes():
    """Verify non-bytes content raises FileValidationError."""
    with pytest.raises(FileValidationError):
        validate_upload("test.txt", "not bytes")  # type: ignore[arg-type]


# ==============================================================================
# 3. Text Extraction Tests (PyMuPDF, 1-based page numbers, empty/image-only PDFs)
# ==============================================================================


def test_extract_text_markdown():
    """Verify Markdown text extraction produces single ExtractedPage with page_number=None."""
    md_content = b"# Architecture\n\nSingle process FastAPI server with SQLite."
    upload = validate_upload("arch.md", md_content)
    extracted = extract_text(upload)

    assert isinstance(extracted, ExtractedDocument)
    assert extracted.filename == "arch.md"
    assert extracted.file_type == "md"
    assert extracted.content_sha256 == upload.content_sha256
    assert len(extracted.pages) == 1
    assert extracted.pages[0].page_number is None
    assert "Single process FastAPI server" in extracted.pages[0].text
    assert extracted.total_characters == len(extracted.pages[0].text)


def test_extract_text_txt():
    """Verify TXT text extraction produces single ExtractedPage with page_number=None."""
    txt_content = b"Line 1\nLine 2\nLine 3"
    upload = validate_upload("lines.txt", txt_content)
    extracted = extract_text(upload)

    assert len(extracted.pages) == 1
    assert extracted.pages[0].page_number is None
    assert extracted.pages[0].text == "Line 1\nLine 2\nLine 3"


def test_extract_text_pdf_multipage_numbering():
    """Verify multi-page PDF extraction preserves 1-based physical page numbers."""
    page_texts = [
        "First page content with overview.",
        "Second page details and architecture.",
        "Third page troubleshooting and FAQ.",
    ]
    pdf_bytes = _create_minimal_pdf(page_texts)
    upload = validate_upload("multi.pdf", pdf_bytes)
    extracted = extract_text(upload)

    assert len(extracted.pages) == 3
    assert extracted.pages[0].page_number == 1
    assert "First page" in extracted.pages[0].text
    assert extracted.pages[1].page_number == 2
    assert "Second page" in extracted.pages[1].text
    assert extracted.pages[2].page_number == 3
    assert "Third page" in extracted.pages[2].text


def test_extract_text_pdf_with_interleaved_empty_page():
    """Verify PDF with an empty page preserves correct 1-based page numbers for non-empty pages."""
    # Page 1 has text, Page 2 is blank (""), Page 3 has text
    page_texts = ["Page One Text", "", "Page Three Text"]
    pdf_bytes = _create_minimal_pdf(page_texts)
    upload = validate_upload("sparse.pdf", pdf_bytes)
    extracted = extract_text(upload)

    assert len(extracted.pages) == 2
    assert extracted.pages[0].page_number == 1
    assert "Page One" in extracted.pages[0].text
    assert extracted.pages[1].page_number == 3
    assert "Page Three" in extracted.pages[1].text


def test_extract_text_empty_pdf():
    """Verify PDF with 0 extractable text and no images raises EmptyExtractionError."""
    pdf_bytes = _create_minimal_pdf(["", "   \n\t  "])
    upload = validate_upload("empty_pages.pdf", pdf_bytes)
    with pytest.raises(EmptyExtractionError) as exc_info:
        extract_text(upload)
    assert "no extractable text" in str(exc_info.value)


def test_extract_text_image_only_pdf():
    """Verify image-only PDF raises ImageOnlyPDFError clearly noting OCR is not supported."""
    pdf_bytes = _create_image_only_pdf()
    upload = validate_upload("scanned.pdf", pdf_bytes)
    with pytest.raises(ImageOnlyPDFError) as exc_info:
        extract_text(upload)
    assert "only images and no extractable text; OCR is not supported" in str(exc_info.value)


def test_extract_text_whitespace_only_markdown():
    """Verify Markdown with only whitespace raises EmptyExtractionError."""
    upload = validate_upload("blank.md", b"   \n\n\t  ")
    with pytest.raises(EmptyExtractionError) as exc_info:
        extract_text(upload)
    assert "no extractable text" in str(exc_info.value)


def test_extract_text_whitespace_only_txt():
    """Verify TXT with only whitespace raises EmptyExtractionError."""
    upload = validate_upload("blank.txt", b"\n\r\t   ")
    with pytest.raises(EmptyExtractionError) as exc_info:
        extract_text(upload)
    assert "no extractable text" in str(exc_info.value)


# ==============================================================================
# 4. Deterministic Chunking Tests (Boundaries, Overlap, Indexes, Page Metadata)
# ==============================================================================


def test_chunk_text_empty():
    """Verify chunk_text on empty string returns empty list."""
    assert chunk_text("") == []


def test_chunk_text_smaller_than_chunk_size():
    """Verify chunk_text on text smaller than chunk_size returns single chunk."""
    text = "Short text"
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)
    assert chunks == [text]


def test_chunk_text_exact_chunk_size():
    """Verify chunk_text on text exactly equal to chunk_size returns single chunk."""
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)
    assert chunks == [text]


def test_chunk_text_deterministic_boundaries_and_exact_overlap():
    """Verify chunk_text produces exact chunk lengths and exact overlap."""
    chunk_size = 1000
    chunk_overlap = 150
    step = chunk_size - chunk_overlap  # 850

    # 2500 character string
    text = "".join(f"{i % 10}" for i in range(2500))
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Expected:
    # Chunk 0: [0:1000], len=1000
    # Chunk 1: [850:1850], len=1000
    # Chunk 2: [1700:2500], len=800
    assert len(chunks) == 3
    assert chunks[0] == text[0:1000]
    assert chunks[1] == text[850:1850]
    assert chunks[2] == text[1700:2500]

    # Verify overlap between chunk 0 and chunk 1
    assert chunks[0][-chunk_overlap:] == chunks[1][:chunk_overlap]
    # Verify overlap between chunk 1 and chunk 2
    assert chunks[1][-chunk_overlap:] == chunks[2][:chunk_overlap]


def test_chunk_text_zero_overlap():
    """Verify chunk_text with 0 overlap produces contiguous non-overlapping partitions."""
    text = "0123456789ABCDEF"
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=0)

    assert chunks == ["01234", "56789", "ABCDE", "F"]
    assert "".join(chunks) == text


@pytest.mark.parametrize(
    "invalid_size, invalid_overlap",
    [
        (0, 150),
        (-10, 50),
        (100, -1),
        (100, 100),  # overlap == size
        (100, 150),  # overlap > size
    ],
)
def test_chunk_text_invalid_parameters(invalid_size: int, invalid_overlap: int):
    """Verify invalid chunking parameters raise ChunkingError / ValueError."""
    with pytest.raises((ChunkingError, ValueError)):
        chunk_text("Some text", chunk_size=invalid_size, chunk_overlap=invalid_overlap)


def test_chunk_document_pdf_preserves_page_boundaries():
    """Verify PDF chunking never crosses page boundaries and preserves global chunk_index."""
    # Page 1: 1500 chars -> 2 chunks (size=1000, overlap=150)
    # Page 2: 400 chars -> 1 chunk
    # Page 3: 1100 chars -> 2 chunks
    page1_text = "A" * 1500
    page2_text = "B" * 400
    page3_text = "C" * 1100

    extracted_pages = [
        ExtractedPage(text=page1_text, page_number=1),
        ExtractedPage(text=page2_text, page_number=2),
        ExtractedPage(text=page3_text, page_number=3),
    ]
    extracted_doc = ExtractedDocument(
        filename="test.pdf",
        file_type="pdf",
        content_sha256="fake_sha",
        pages=extracted_pages,
    )

    doc_id = str(uuid.uuid4())
    chunks = chunk_document(
        extracted_doc, chunk_size=1000, chunk_overlap=150, document_id=doc_id
    )

    # 2 + 1 + 2 = 5 chunks total
    assert len(chunks) == 5

    # Verify global chunk indexing
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.document_id == doc_id
        assert c.chunk_id == f"{doc_id}:{idx}"

    # Verify Page 1 chunks
    assert chunks[0].page_number == 1
    assert chunks[0].text == "A" * 1000
    assert chunks[1].page_number == 1
    assert chunks[1].text == "A" * 650

    # Verify Page 2 chunk (boundary invariant: no 'A' or 'C' in Page 2 chunk!)
    assert chunks[2].page_number == 2
    assert chunks[2].text == "B" * 400

    # Verify Page 3 chunks (boundary invariant: no 'B' in Page 3 chunks!)
    assert chunks[3].page_number == 3
    assert chunks[3].text == "C" * 1000
    assert chunks[4].page_number == 3
    assert chunks[4].text == "C" * 250


def test_chunk_document_markdown_page_number_none():
    """Verify Markdown chunking sets page_number to None for all chunks."""
    text = "Markdown content. " * 100
    extracted = ExtractedDocument(
        filename="guide.md",
        file_type="md",
        content_sha256="abc123",
        pages=[ExtractedPage(text=text, page_number=None)],
    )

    chunks = chunk_document(extracted, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_number is None


def test_chunk_document_empty_pages():
    """Verify chunk_document with empty pages returns empty chunk list."""
    assert chunk_document([], chunk_size=1000, chunk_overlap=150) == []


# ==============================================================================
# 5. Absence of Persisted Upload Artifacts
# ==============================================================================


def test_absence_of_persisted_upload_artifacts():
    """Verify that validating, extracting, and chunking never writes files to disk."""
    data_dir = pathlib.Path("./data")
    tmp_dir = pathlib.Path(tempfile.gettempdir())

    def get_dir_state(p: pathlib.Path) -> set[str]:
        if not p.exists():
            return set()
        return {str(f) for f in p.rglob("*") if f.is_file()}

    data_before = get_dir_state(data_dir)
    tmp_before = get_dir_state(tmp_dir)

    # Process PDF
    pdf_bytes = _create_minimal_pdf(["Ephemeral PDF text on page 1", "Ephemeral text on page 2"])
    upload_pdf = validate_upload("ephemeral.pdf", pdf_bytes)
    extracted_pdf = extract_text(upload_pdf)
    chunks_pdf = chunk_document(extracted_pdf)
    assert len(chunks_pdf) == 2

    # Process Markdown
    md_bytes = b"# Ephemeral Markdown\n\nNo disk files should be created."
    upload_md = validate_upload("ephemeral.md", md_bytes)
    extracted_md = extract_text(upload_md)
    chunks_md = chunk_document(extracted_md)
    assert len(chunks_md) == 1

    # Process TXT
    txt_bytes = b"Ephemeral text payload."
    upload_txt = validate_upload("ephemeral.txt", txt_bytes)
    extracted_txt = extract_text(upload_txt)
    chunks_txt = chunk_document(extracted_txt)
    assert len(chunks_txt) == 1

    data_after = get_dir_state(data_dir)
    tmp_after = get_dir_state(tmp_dir)

    # Assert no new files in data_dir
    new_data_files = data_after - data_before
    assert new_data_files == set(), f"Persisted files found in data_dir: {new_data_files}"

    # Assert no ephemeral upload files leaked into tmp
    new_tmp_files = {f for f in (tmp_after - tmp_before) if "ephemeral" in f}
    assert new_tmp_files == set(), f"Persisted upload files found in tmp: {new_tmp_files}"


# ==============================================================================
# 6. Sample Corpus Ingestion Tests
# ==============================================================================


@pytest.mark.parametrize(
    "sample_rel_path",
    [
        "sample_docs/quickstart.md",
        "sample_docs/api-reference.md",
        "sample_docs/troubleshooting.md",
    ],
)
def test_sample_docs_validation_extraction_chunking(sample_rel_path: str):
    """Verify all repository sample docs pass validation, extraction, and chunking."""
    file_path = pathlib.Path(sample_rel_path)
    assert file_path.exists(), f"Sample doc {sample_rel_path} must exist"

    raw_bytes = file_path.read_bytes()
    upload = validate_upload(file_path.name, raw_bytes)

    assert upload.filename == file_path.name
    assert upload.file_type == "md"
    assert upload.content_sha256 == hashlib.sha256(raw_bytes).hexdigest()

    extracted = extract_text(upload)
    assert len(extracted.pages) == 1
    assert extracted.pages[0].page_number is None
    assert len(extracted.pages[0].text) > 0

    settings = get_settings()
    chunks = chunk_document(extracted, document_id="sample-uuid")
    assert len(chunks) > 0

    # Verify all chunk boundaries and properties
    for chunk in chunks:
        assert chunk.chunk_id.startswith("sample-uuid:")
        assert chunk.page_number is None
        assert len(chunk.text) <= settings.CHUNK_SIZE
