"""Comprehensive tests for the persistent ChromaDB vector-store adapter."""

import math
import uuid
import pytest
from starlette.testclient import TestClient

from app.core.config import Settings
from app.db.chroma import (
    DEFAULT_COLLECTION_NAME,
    HNSW_SPACE_COSINE,
    ChromaAdapter,
    ChromaStorageError,
    ChromaValidationError,
    ChunkRecord,
    check_chroma_readiness,
    format_chunk_id,
)
from app.db.sqlite import init_db
from app.main import create_app
from app.models.chunk import DocumentChunk


def test_deterministic_chunk_id_formatting():
    """Verify chunk ID is deterministically formatted as <document_uuid>:<chunk_index>."""
    doc_id = str(uuid.uuid4())
    assert format_chunk_id(doc_id, 0) == f"{doc_id}:0"
    assert format_chunk_id(doc_id, 42) == f"{doc_id}:42"

    chunk = ChunkRecord(
        document_id=doc_id,
        chunk_index=3,
        text="Sample text",
        embedding=[0.1, 0.2],
        filename="test.txt",
        file_type="txt",
    )
    assert chunk.chunk_id == f"{doc_id}:3"


def test_chunk_record_from_document_chunk():
    """Verify ChunkRecord can be instantiated from DocumentChunk."""
    doc_id = str(uuid.uuid4())
    doc_chunk = DocumentChunk(
        chunk_index=2,
        text="Extracted text chunk",
        page_number=5,
        document_id=doc_id,
    )

    record = ChunkRecord.from_document_chunk(
        chunk=doc_chunk,
        embedding=[0.1] * 4,
        filename="guide.pdf",
        file_type="pdf",
    )

    assert record.document_id == doc_id
    assert record.chunk_index == 2
    assert record.page_number == 5
    assert record.filename == "guide.pdf"
    assert record.file_type == "pdf"
    assert record.chunk_id == f"{doc_id}:2"


def test_chunk_record_from_document_chunk_missing_id():
    """Verify constructing ChunkRecord from DocumentChunk without document_id raises ChromaValidationError."""
    doc_chunk = DocumentChunk(
        chunk_index=0,
        text="Text",
        document_id=None,
    )
    with pytest.raises(ChromaValidationError):
        ChunkRecord.from_document_chunk(
            chunk=doc_chunk,
            embedding=[0.1] * 4,
            filename="guide.pdf",
            file_type="pdf",
        )


def test_collection_configuration_and_cosine_distance(tmp_path):
    """Verify collection name is devguide_chunks and cosine distance is configured explicitly."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    adapter.init_collection()

    col = adapter._get_collection()
    assert col.name == DEFAULT_COLLECTION_NAME
    assert col.metadata is not None
    assert col.metadata.get("hnsw:space") == HNSW_SPACE_COSINE
    # Silent model selection is disabled
    assert col._embedding_function is None


def test_metadata_handling_pdf_vs_markdown_txt(tmp_path):
    """Verify PDF chunks store page_number while Markdown and TXT chunks omit page_number from Chroma metadata."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id = str(uuid.uuid4())

    pdf_chunk = ChunkRecord(
        document_id=doc_id,
        chunk_index=0,
        text="PDF page chunk",
        embedding=[0.1, 0.2, 0.3],
        filename="manual.pdf",
        file_type="pdf",
        page_number=1,
    )
    md_chunk = ChunkRecord(
        document_id=doc_id,
        chunk_index=1,
        text="Markdown chunk without page",
        embedding=[0.4, 0.5, 0.6],
        filename="README.md",
        file_type="md",
        page_number=None,
    )
    txt_chunk = ChunkRecord(
        document_id=doc_id,
        chunk_index=2,
        text="TXT chunk without page",
        embedding=[0.7, 0.8, 0.9],
        filename="notes.txt",
        file_type="txt",
        page_number=None,
    )

    adapter.add_chunks([pdf_chunk, md_chunk, txt_chunk])

    raw_col = adapter._get_collection()
    raw_results = raw_col.get(ids=[pdf_chunk.chunk_id, md_chunk.chunk_id, txt_chunk.chunk_id])
    metas_by_id = dict(zip(raw_results["ids"], raw_results["metadatas"]))

    # PDF metadata must contain page_number as integer
    assert "page_number" in metas_by_id[pdf_chunk.chunk_id]
    assert metas_by_id[pdf_chunk.chunk_id]["page_number"] == 1
    assert metas_by_id[pdf_chunk.chunk_id]["document_id"] == doc_id
    assert metas_by_id[pdf_chunk.chunk_id]["filename"] == "manual.pdf"
    assert metas_by_id[pdf_chunk.chunk_id]["file_type"] == "pdf"
    assert metas_by_id[pdf_chunk.chunk_id]["chunk_index"] == 0

    # MD and TXT metadata must omit page_number
    assert "page_number" not in metas_by_id[md_chunk.chunk_id]
    assert metas_by_id[md_chunk.chunk_id]["filename"] == "README.md"
    assert "page_number" not in metas_by_id[txt_chunk.chunk_id]
    assert metas_by_id[txt_chunk.chunk_id]["filename"] == "notes.txt"

    # Adapter's get_records_by_document should map omitted page_number to None
    records = adapter.get_records_by_document(doc_id)
    assert len(records) == 3
    assert records[0].page_number == 1
    assert records[1].page_number is None
    assert records[2].page_number is None


def test_cardinality_and_schema_validation_before_writes(tmp_path):
    """Verify pre-write validation for invalid inputs, dimensions, and duplicate IDs."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id = str(uuid.uuid4())

    # Empty document_id
    with pytest.raises(ChromaValidationError):
        adapter.add_chunks([
            ChunkRecord(
                document_id="",
                chunk_index=0,
                text="Text",
                embedding=[0.1, 0.2],
                filename="a.txt",
                file_type="txt",
            )
        ])

    # Negative chunk_index
    with pytest.raises(ChromaValidationError):
        adapter.add_chunks([
            ChunkRecord(
                document_id=doc_id,
                chunk_index=-1,
                text="Text",
                embedding=[0.1, 0.2],
                filename="a.txt",
                file_type="txt",
            )
        ])

    # Empty text
    with pytest.raises(ChromaValidationError):
        adapter.add_chunks([
            ChunkRecord(
                document_id=doc_id,
                chunk_index=0,
                text="   ",
                embedding=[0.1, 0.2],
                filename="a.txt",
                file_type="txt",
            )
        ])

    # Non-finite vector element (NaN)
    with pytest.raises(ChromaValidationError):
        adapter.add_chunks([
            ChunkRecord(
                document_id=doc_id,
                chunk_index=0,
                text="Text",
                embedding=[0.1, float("nan")],
                filename="a.txt",
                file_type="txt",
            )
        ])

    # Non-finite vector element (Infinity)
    with pytest.raises(ChromaValidationError):
        adapter.add_chunks([
            ChunkRecord(
                document_id=doc_id,
                chunk_index=0,
                text="Text",
                embedding=[0.1, float("inf")],
                filename="a.txt",
                file_type="txt",
            )
        ])

    # Vector dimension mismatch across chunks in the same batch
    with pytest.raises(ChromaValidationError) as exc_info:
        adapter.add_chunks([
            ChunkRecord(
                document_id=doc_id,
                chunk_index=0,
                text="Chunk 0",
                embedding=[0.1, 0.2, 0.3],
                filename="a.txt",
                file_type="txt",
            ),
            ChunkRecord(
                document_id=doc_id,
                chunk_index=1,
                text="Chunk 1",
                embedding=[0.1, 0.2],  # 2 dims instead of 3
                filename="a.txt",
                file_type="txt",
            ),
        ])
    assert "vector dimension" in str(exc_info.value)

    # Duplicate chunk IDs in the same batch
    with pytest.raises(ChromaValidationError) as exc_info:
        adapter.add_chunks([
            ChunkRecord(
                document_id=doc_id,
                chunk_index=0,
                text="Chunk 0",
                embedding=[0.1, 0.2],
                filename="a.txt",
                file_type="txt",
            ),
            ChunkRecord(
                document_id=doc_id,
                chunk_index=0,  # Duplicate chunk_index for same document_id
                text="Chunk 0 duplicate",
                embedding=[0.3, 0.4],
                filename="a.txt",
                file_type="txt",
            ),
        ])
    assert "Duplicate chunk IDs" in str(exc_info.value)


def test_empty_chunks_write_no_op(tmp_path):
    """Verify passing an empty list to add_chunks or upsert_chunks is a safe no-op."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    adapter.add_chunks([])
    adapter.upsert_chunks([])
    assert not (tmp_path / "chroma").exists()


def test_upsert_chunks_updates_existing_records(tmp_path):
    """Verify upsert_chunks overwrites existing chunk content and embedding."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id = str(uuid.uuid4())

    initial_chunk = ChunkRecord(
        document_id=doc_id,
        chunk_index=0,
        text="Original text",
        embedding=[1.0, 0.0],
        filename="test.txt",
        file_type="txt",
    )
    adapter.add_chunks([initial_chunk])
    assert adapter.count() == 1

    updated_chunk = ChunkRecord(
        document_id=doc_id,
        chunk_index=0,
        text="Updated text content",
        embedding=[0.0, 1.0],
        filename="test.txt",
        file_type="txt",
    )
    adapter.upsert_chunks([updated_chunk])
    assert adapter.count() == 1

    records = adapter.get_records_by_document(doc_id, include_embeddings=True)
    assert len(records) == 1
    assert records[0].text == "Updated text content"
    assert records[0].embedding == [0.0, 1.0]


def test_query_nearest_mapping_and_cosine_distance(tmp_path):
    """Verify nearest chunk query returns VectorSearchResult with correct cosine distance and ordering."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id = str(uuid.uuid4())

    # Create 3 chunks with distinct vectors in 2D space
    chunk_identical = ChunkRecord(
        document_id=doc_id,
        chunk_index=0,
        text="Identical vector text",
        embedding=[1.0, 0.0],
        filename="doc.txt",
        file_type="txt",
    )
    chunk_orthogonal = ChunkRecord(
        document_id=doc_id,
        chunk_index=1,
        text="Orthogonal vector text",
        embedding=[0.0, 1.0],
        filename="doc.txt",
        file_type="txt",
    )
    chunk_opposite = ChunkRecord(
        document_id=doc_id,
        chunk_index=2,
        text="Opposite vector text",
        embedding=[-1.0, 0.0],
        filename="doc.txt",
        file_type="txt",
    )

    adapter.add_chunks([chunk_opposite, chunk_orthogonal, chunk_identical])

    # Query with [1.0, 0.0]
    results = adapter.query_nearest(query_vector=[1.0, 0.0], top_k=3)

    assert len(results) == 3
    # First result should be identical chunk (distance ≈ 0.0)
    assert results[0].id == chunk_identical.chunk_id
    assert results[0].chunk_index == 0
    assert results[0].text == "Identical vector text"
    assert pytest.approx(results[0].distance, abs=1e-4) == 0.0

    # Second result should be orthogonal chunk (distance ≈ 1.0)
    assert results[1].id == chunk_orthogonal.chunk_id
    assert pytest.approx(results[1].distance, abs=1e-4) == 1.0

    # Third result should be opposite chunk (distance ≈ 2.0)
    assert results[2].id == chunk_opposite.chunk_id
    assert pytest.approx(results[2].distance, abs=1e-4) == 2.0


def test_query_nearest_top_k_bounds_and_empty_collection(tmp_path):
    """Verify top_k limit is respected and query against empty collection returns empty list."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    adapter.init_collection()

    # Empty collection query returns []
    empty_results = adapter.query_nearest(query_vector=[0.1, 0.2], top_k=5)
    assert empty_results == []

    # Insert 5 chunks
    doc_id = str(uuid.uuid4())
    chunks = [
        ChunkRecord(
            document_id=doc_id,
            chunk_index=i,
            text=f"Chunk {i}",
            embedding=[0.1 * i, 0.2],
            filename="a.txt",
            file_type="txt",
        )
        for i in range(5)
    ]
    adapter.add_chunks(chunks)

    # Query with top_k=2
    top2 = adapter.query_nearest(query_vector=[0.0, 0.2], top_k=2)
    assert len(top2) == 2

    # Query with top_k > count
    top10 = adapter.query_nearest(query_vector=[0.0, 0.2], top_k=10)
    assert len(top10) == 5


def test_query_nearest_validation(tmp_path):
    """Verify query_nearest validates vector and top_k parameters."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")

    with pytest.raises(ChromaValidationError):
        adapter.query_nearest(query_vector=[], top_k=5)

    with pytest.raises(ChromaValidationError):
        adapter.query_nearest(query_vector=[0.1, float("nan")], top_k=5)

    with pytest.raises(ChromaValidationError):
        adapter.query_nearest(query_vector=[0.1, 0.2], top_k=0)

    with pytest.raises(ChromaValidationError):
        adapter.query_nearest(query_vector=[0.1, 0.2], top_k=-5)


def test_get_records_by_document(tmp_path):
    """Verify get_records_by_document returns all chunks for that document sorted by chunk_index."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id1 = str(uuid.uuid4())
    doc_id2 = str(uuid.uuid4())

    # Add chunks for doc1 out of order
    chunks_doc1 = [
        ChunkRecord(document_id=doc_id1, chunk_index=2, text="Doc1 chunk 2", embedding=[0.3]*4, filename="1.txt", file_type="txt"),
        ChunkRecord(document_id=doc_id1, chunk_index=0, text="Doc1 chunk 0", embedding=[0.1]*4, filename="1.txt", file_type="txt"),
        ChunkRecord(document_id=doc_id1, chunk_index=1, text="Doc1 chunk 1", embedding=[0.2]*4, filename="1.txt", file_type="txt"),
    ]
    chunks_doc2 = [
        ChunkRecord(document_id=doc_id2, chunk_index=0, text="Doc2 chunk 0", embedding=[0.9]*4, filename="2.txt", file_type="txt"),
    ]

    adapter.add_chunks(chunks_doc1 + chunks_doc2)

    records = adapter.get_records_by_document(doc_id1)
    assert len(records) == 3
    # Verified sorted order
    assert [r.chunk_index for r in records] == [0, 1, 2]
    assert [r.text for r in records] == ["Doc1 chunk 0", "Doc1 chunk 1", "Doc1 chunk 2"]

    # Nonexistent document returns []
    nonexistent = adapter.get_records_by_document(str(uuid.uuid4()))
    assert nonexistent == []


def test_delete_records_by_document(tmp_path):
    """Verify delete_records_by_document deletes all chunks of the target document only."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id1 = str(uuid.uuid4())
    doc_id2 = str(uuid.uuid4())

    chunks = [
        ChunkRecord(document_id=doc_id1, chunk_index=0, text="Doc1 c0", embedding=[0.1]*4, filename="1.txt", file_type="txt"),
        ChunkRecord(document_id=doc_id1, chunk_index=1, text="Doc1 c1", embedding=[0.2]*4, filename="1.txt", file_type="txt"),
        ChunkRecord(document_id=doc_id2, chunk_index=0, text="Doc2 c0", embedding=[0.3]*4, filename="2.txt", file_type="txt"),
    ]
    adapter.add_chunks(chunks)
    assert adapter.count() == 3

    # Delete doc1
    deleted_count = adapter.delete_records_by_document(doc_id1)
    assert deleted_count == 2
    assert adapter.count() == 1

    # Doc1 records should be gone
    assert adapter.get_records_by_document(doc_id1) == []
    # Doc2 records remain intact
    doc2_records = adapter.get_records_by_document(doc_id2)
    assert len(doc2_records) == 1
    assert doc2_records[0].text == "Doc2 c0"

    # Deleting nonexistent document returns 0
    assert adapter.delete_records_by_document(str(uuid.uuid4())) == 0


def test_delete_records_by_document_when_chroma_delete_returns_none(tmp_path, monkeypatch):
    """Verify adapter returns correct count even when collection.delete() returns None."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id = str(uuid.uuid4())
    chunks = [
        ChunkRecord(document_id=doc_id, chunk_index=0, text="c0", embedding=[0.1]*4, filename="1.txt", file_type="txt"),
        ChunkRecord(document_id=doc_id, chunk_index=1, text="c1", embedding=[0.2]*4, filename="1.txt", file_type="txt"),
    ]
    adapter.add_chunks(chunks)
    assert adapter.count() == 2

    col = adapter._get_collection()
    real_delete = col.delete

    def fake_delete(*args, **kwargs):
        real_delete(*args, **kwargs)
        return None

    monkeypatch.setattr(col, "delete", fake_delete)

    deleted_count = adapter.delete_records_by_document(doc_id)
    assert deleted_count == 2
    assert adapter.count() == 0


def test_delete_records_by_document_storage_error_preservation(tmp_path, monkeypatch):
    """Verify ChromaStorageError is raised if underlying get or delete fails during deletion."""
    adapter = ChromaAdapter(chroma_path=tmp_path / "chroma")
    doc_id = str(uuid.uuid4())
    adapter.add_chunks([
        ChunkRecord(document_id=doc_id, chunk_index=0, text="c0", embedding=[0.1]*4, filename="1.txt", file_type="txt"),
    ])
    col = adapter._get_collection()

    # Case 1: col.get fails
    def broken_get(*args, **kwargs):
        raise RuntimeError("Disk read error")

    monkeypatch.setattr(col, "get", broken_get)
    with pytest.raises(ChromaStorageError) as exc1:
        adapter.delete_records_by_document(doc_id)
    assert "Failed to delete chunks" in str(exc1.value)

    # Case 2: col.delete fails
    monkeypatch.undo()
    def broken_delete(*args, **kwargs):
        raise RuntimeError("Write lock error")

    monkeypatch.setattr(col, "delete", broken_delete)
    with pytest.raises(ChromaStorageError) as exc2:
        adapter.delete_records_by_document(doc_id)
    assert "Failed to delete chunks" in str(exc2.value)


def test_persistence_after_reopen(tmp_path):
    """Verify data, cosine distance metadata, and chunk queries persist after closing and reopening adapter."""
    chroma_dir = tmp_path / "chroma_persistent"
    doc_id = str(uuid.uuid4())

    # Session 1: Write data
    adapter1 = ChromaAdapter(chroma_path=chroma_dir)
    chunks = [
        ChunkRecord(
            document_id=doc_id,
            chunk_index=0,
            text="First chunk persisted",
            embedding=[0.8, 0.6],
            filename="data.pdf",
            file_type="pdf",
            page_number=1,
        ),
        ChunkRecord(
            document_id=doc_id,
            chunk_index=1,
            text="Second chunk persisted",
            embedding=[0.0, 1.0],
            filename="data.pdf",
            file_type="pdf",
            page_number=2,
        ),
    ]
    adapter1.add_chunks(chunks)
    assert adapter1.count() == 2

    # Drop adapter1
    del adapter1

    # Session 2: Open a fresh adapter instance pointing to the same directory
    adapter2 = ChromaAdapter(chroma_path=chroma_dir)
    assert adapter2.count() == 2
    assert adapter2.check_health() is True

    # Check collection metadata persisted
    col = adapter2._get_collection()
    assert col.metadata.get("hnsw:space") == HNSW_SPACE_COSINE

    # Query vectors in reopened collection
    results = adapter2.query_nearest(query_vector=[0.8, 0.6], top_k=1)
    assert len(results) == 1
    assert results[0].id == f"{doc_id}:0"
    assert results[0].text == "First chunk persisted"
    assert results[0].page_number == 1
    assert pytest.approx(results[0].distance, abs=1e-4) == 0.0

    # Get records in reopened collection
    records = adapter2.get_records_by_document(doc_id)
    assert len(records) == 2
    assert records[0].chunk_index == 0
    assert records[1].chunk_index == 1


def test_readiness_probe_does_not_create_missing_store(tmp_path):
    """Verify check_chroma_readiness does not create missing directory or files as a side effect."""
    missing_dir = tmp_path / "missing_chroma"
    assert not missing_dir.exists()

    status = check_chroma_readiness(missing_dir)

    assert status == "not_initialized"
    # Target directory must NOT be created as a side effect
    assert not missing_dir.exists()


def test_readiness_probe_empty_directory(tmp_path):
    """Verify check_chroma_readiness returns not_initialized for empty directory without creating sqlite file."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    status = check_chroma_readiness(empty_dir)

    assert status == "not_initialized"
    assert not (empty_dir / "chroma.sqlite3").exists()


def test_readiness_probe_non_directory_file(tmp_path):
    """Verify check_chroma_readiness returns error if target path is a regular file."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("Hello")

    status = check_chroma_readiness(file_path)

    assert status == "error"


def test_readiness_probe_corrupt_database(tmp_path):
    """Verify check_chroma_readiness returns error if sqlite file is corrupted."""
    corrupt_dir = tmp_path / "corrupt_chroma"
    corrupt_dir.mkdir()
    (corrupt_dir / "chroma.sqlite3").write_text("garbage content not a valid sqlite db")

    status = check_chroma_readiness(corrupt_dir)

    assert status == "error"


def test_readiness_probe_initialized_collection(tmp_path):
    """Verify check_chroma_readiness returns ok when collection is initialized."""
    chroma_dir = tmp_path / "initialized_chroma"
    adapter = ChromaAdapter(chroma_path=chroma_dir)
    adapter.init_collection()

    status = check_chroma_readiness(chroma_dir)
    assert status == "ok"
    assert adapter.check_health() is True


def test_readyz_fastapi_integration_all_ready(tmp_path):
    """Verify GET /readyz returns 200 OK and ready when SQLite and ChromaDB are both initialized."""
    db_file = tmp_path / "devguide.db"
    chroma_dir = tmp_path / "chroma"

    # Initialize SQLite
    init_db(db_path=db_file, data_dir=tmp_path)
    # Initialize ChromaDB
    adapter = ChromaAdapter(chroma_path=chroma_dir)
    adapter.init_collection()

    settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_file,
        CHROMA_PATH=chroma_dir,
        OPENAI_API_KEY=None,  # Missing API key must not affect readiness
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"] == {
        "sqlite": "ok",
        "chromadb": "ok",
    }
    assert "openai" not in data["checks"]


def test_readyz_fastapi_integration_chroma_not_initialized(tmp_path):
    """Verify GET /readyz returns 503 not_ready when ChromaDB is not initialized."""
    db_file = tmp_path / "devguide.db"
    chroma_dir = tmp_path / "chroma_uninitialized"

    init_db(db_path=db_file, data_dir=tmp_path)

    settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_file,
        CHROMA_PATH=chroma_dir,
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "ok",
        "chromadb": "not_initialized",
    }
    # Storage path must not have been created by calling /readyz
    assert not chroma_dir.exists()
