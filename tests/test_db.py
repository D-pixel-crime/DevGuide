"""Tests for SQLite database foundation."""

import pathlib
import sqlite3
from unittest.mock import MagicMock

import pytest

from app.db.sqlite import (
    CREATE_DOCUMENTS_TABLE_SQL,
    DOCUMENTS_TABLE_NAME,
    check_sqlite_readiness,
    get_connection,
    init_db,
)


def test_first_time_initialization(tmp_path: pathlib.Path) -> None:
    """Verify first-time initialization creates database file and documents table."""
    db_file = tmp_path / "devguide.db"
    assert not db_file.exists()

    init_db(db_path=db_file, data_dir=tmp_path)

    assert db_file.is_file()

    with get_connection(db_file) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (DOCUMENTS_TABLE_NAME,),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["name"] == DOCUMENTS_TABLE_NAME


def test_repeated_initialization(tmp_path: pathlib.Path) -> None:
    """Verify repeated initialization is safe and preserves existing data."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    # Insert a document record
    with get_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("doc-1", "guide.md", "md", "hash_abc", 4, "2026-11-23T00:00:00Z"),
        )

    # Run repeated initialization
    init_db(db_path=db_file, data_dir=tmp_path)

    # Verify the document record still exists untouched
    with get_connection(db_file) as conn:
        cursor = conn.execute("SELECT * FROM documents WHERE id = ?;", ("doc-1",))
        row = cursor.fetchone()
        assert row is not None
        assert row["filename"] == "guide.md"
        assert row["chunk_count"] == 4

        count_row = conn.execute("SELECT COUNT(*) AS total FROM documents;").fetchone()
        assert count_row["total"] == 1


def test_exact_documents_table_schema(tmp_path: pathlib.Path) -> None:
    """Verify exact documents table schema columns, types, nullability, and primary key."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    with get_connection(db_file) as conn:
        cursor = conn.execute("PRAGMA table_info(documents);")
        columns = {row["name"]: dict(row) for row in cursor.fetchall()}

    expected_columns = {
        "id": {"type": "TEXT", "notnull": 0, "pk": 1},  # In SQLite PK implies not null
        "filename": {"type": "TEXT", "notnull": 1, "pk": 0},
        "file_type": {"type": "TEXT", "notnull": 1, "pk": 0},
        "content_sha256": {"type": "TEXT", "notnull": 1, "pk": 0},
        "chunk_count": {"type": "INTEGER", "notnull": 1, "pk": 0},
        "created_at": {"type": "TEXT", "notnull": 1, "pk": 0},
    }

    assert set(columns.keys()) == set(expected_columns.keys())

    for col_name, expected_props in expected_columns.items():
        actual = columns[col_name]
        assert actual["type"] == expected_props["type"]
        assert actual["notnull"] == expected_props["notnull"]
        assert actual["pk"] == expected_props["pk"]

    # Verify CHECK constraint is present in schema DDL
    with get_connection(db_file) as conn:
        schema_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents';"
        ).fetchone()
        assert "CHECK(chunk_count >= 0)" in schema_row["sql"]


def test_exactly_one_application_table(tmp_path: pathlib.Path) -> None:
    """Verify exactly one application table named documents exists."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    with get_connection(db_file) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row["name"] for row in cursor.fetchall()]

    assert tables == ["documents"]


def test_non_negative_chunk_count_constraint(tmp_path: pathlib.Path) -> None:
    """Verify CHECK(chunk_count >= 0) constraint rejects negative counts and allows zero/positive."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    # Negative chunk count must fail
    with pytest.raises(sqlite3.IntegrityError):
        with get_connection(db_file) as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                ("doc-neg", "neg.md", "md", "hash_neg", -1, "2026-11-23T00:00:00Z"),
            )

    # Zero chunk count (boundary) must succeed
    with get_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("doc-zero", "zero.md", "md", "hash_zero", 0, "2026-11-23T00:00:00Z"),
        )

    # Positive chunk count must succeed
    with get_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("doc-pos", "pos.md", "md", "hash_pos", 10, "2026-11-23T00:00:00Z"),
        )

    with get_connection(db_file) as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM documents;").fetchone()["c"]
        assert total == 2


def test_unique_content_sha256_constraint(tmp_path: pathlib.Path) -> None:
    """Verify content_sha256 UNIQUE constraint rejects duplicate file hashes."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    with get_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("doc-1", "file1.txt", "txt", "duplicate_hash", 1, "2026-11-23T00:00:00Z"),
        )

    with pytest.raises(sqlite3.IntegrityError):
        with get_connection(db_file) as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                ("doc-2", "file2.txt", "txt", "duplicate_hash", 2, "2026-11-23T00:00:00Z"),
            )


def test_persistence_across_separate_connections(tmp_path: pathlib.Path) -> None:
    """Verify data committed in one connection persists and is visible in separate connections."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    with get_connection(db_file) as conn1:
        conn1.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("persist-1", "doc.pdf", "pdf", "sha_persist", 7, "2026-11-23T01:00:00Z"),
        )

    with get_connection(db_file) as conn2:
        row = conn2.execute(
            "SELECT * FROM documents WHERE id = ?;", ("persist-1",)
        ).fetchone()
        assert row is not None
        assert row["filename"] == "doc.pdf"
        assert row["file_type"] == "pdf"
        assert row["content_sha256"] == "sha_persist"
        assert row["chunk_count"] == 7
        assert row["created_at"] == "2026-11-23T01:00:00Z"


def test_row_mapping_using_sqlite3_row(tmp_path: pathlib.Path) -> None:
    """Verify sqlite3.Row row_factory enables column access by name, index, and keys."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    with get_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("row-1", "mapped.md", "md", "sha_mapped", 3, "2026-11-23T02:00:00Z"),
        )

    with get_connection(db_file) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?;", ("row-1",)).fetchone()
        assert isinstance(row, sqlite3.Row)

        # Access by name
        assert row["id"] == "row-1"
        assert row["filename"] == "mapped.md"
        assert row["file_type"] == "md"
        assert row["content_sha256"] == "sha_mapped"
        assert row["chunk_count"] == 3
        assert row["created_at"] == "2026-11-23T02:00:00Z"

        # Access by index
        assert row[0] == "row-1"
        assert row[1] == "mapped.md"

        # Keys
        assert list(row.keys()) == [
            "id",
            "filename",
            "file_type",
            "content_sha256",
            "chunk_count",
            "created_at",
        ]


def test_foreign_key_enforcement_enabled(tmp_path: pathlib.Path) -> None:
    """Verify PRAGMA foreign_keys = ON is executed and actively enforced on every connection."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    with get_connection(db_file) as conn:
        fk_status = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk_status == 1

        # Create temporary parent and child tables to verify active enforcement
        conn.execute("CREATE TABLE parent_test (id TEXT PRIMARY KEY);")
        conn.execute(
            "CREATE TABLE child_test (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES parent_test(id));"
        )

    # Attempting to insert a child record referencing non-existent parent must fail
    with pytest.raises(sqlite3.IntegrityError):
        with get_connection(db_file) as conn:
            conn.execute("INSERT INTO child_test (id, parent_id) VALUES ('c1', 'p_missing');")


def test_select_1_readiness_check(tmp_path: pathlib.Path) -> None:
    """Verify check_sqlite_readiness executes SELECT 1 and returns 'ok' for healthy database."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    status = check_sqlite_readiness(db_file)
    assert status == "ok"


def test_missing_uninitialized_database_readiness(tmp_path: pathlib.Path) -> None:
    """Verify check_sqlite_readiness returns 'not_initialized' without creating database file."""
    db_file = tmp_path / "uninitialized.db"
    assert not db_file.exists()

    status = check_sqlite_readiness(db_file)
    assert status == "not_initialized"

    # Crucial invariant: readiness check must never create database files on disk
    assert not db_file.exists()


def test_connection_cleanup_on_success(tmp_path: pathlib.Path) -> None:
    """Verify database connection is closed after context manager exits normally."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    captured_conn = None
    with get_connection(db_file) as conn:
        captured_conn = conn
        assert conn.execute("SELECT 1;").fetchone()[0] == 1

    assert captured_conn is not None
    # Operating on a closed SQLite connection raises ProgrammingError
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        captured_conn.execute("SELECT 1;")


def test_connection_cleanup_on_exception(tmp_path: pathlib.Path) -> None:
    """Verify database connection is closed after context manager raises an exception."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    captured_conn = None
    with pytest.raises(RuntimeError, match="intentional failure"):
        with get_connection(db_file) as conn:
            captured_conn = conn
            raise RuntimeError("intentional failure")

    assert captured_conn is not None
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        captured_conn.execute("SELECT 1;")


def test_transaction_rollback_after_failure(tmp_path: pathlib.Path) -> None:
    """Verify failed transactions are rolled back and uncommitted changes are discarded."""
    db_file = tmp_path / "devguide.db"
    init_db(db_path=db_file, data_dir=tmp_path)

    # Initial committed record
    with get_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("doc-committed", "base.txt", "txt", "hash_base", 1, "2026-11-23T00:00:00Z"),
        )

    # Attempt to insert second record but raise an exception before block completion
    with pytest.raises(ValueError, match="simulated error"):
        with get_connection(db_file) as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, file_type, content_sha256, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                ("doc-aborted", "aborted.txt", "txt", "hash_abort", 2, "2026-11-23T00:00:00Z"),
            )
            raise ValueError("simulated error")

    # In a new connection, verify second record was rolled back
    with get_connection(db_file) as conn:
        committed = conn.execute(
            "SELECT * FROM documents WHERE id = ?;", ("doc-committed",)
        ).fetchone()
        assert committed is not None

        aborted = conn.execute(
            "SELECT * FROM documents WHERE id = ?;", ("doc-aborted",)
        ).fetchone()
        assert aborted is None

        count = conn.execute("SELECT COUNT(*) AS total FROM documents;").fetchone()["total"]
        assert count == 1


def test_creation_of_parent_directories_when_requested(tmp_path: pathlib.Path) -> None:
    """Verify init_db creates nested parent directories when explicitly requested."""
    deep_db = tmp_path / "nested" / "deeply" / "storage" / "devguide.db"
    deep_data = tmp_path / "nested" / "deeply" / "storage"

    assert not deep_data.exists()
    assert not deep_db.parent.exists()

    init_db(db_path=deep_db, data_dir=deep_data)

    assert deep_data.is_dir()
    assert deep_db.parent.is_dir()
    assert deep_db.is_file()

    with get_connection(deep_db) as conn:
        assert conn.execute("SELECT 1;").fetchone()[0] == 1


def test_sqlite_readiness_error_on_invalid_target(tmp_path: pathlib.Path) -> None:
    """Verify check_sqlite_readiness returns 'error' when opening the database fails."""
    # Point db_path to a directory instead of a file
    dir_target = tmp_path / "is_a_directory.db"
    dir_target.mkdir()

    # In SQLite, opening a directory as a database file raises OperationalError
    status = check_sqlite_readiness(dir_target)
    assert status == "error"


def test_init_db_without_get_settings_when_arguments_supplied(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify init_db initializes without calling get_settings when db_path and data_dir are supplied."""
    def broken_get_settings():
        raise RuntimeError("get_settings should not be called when both arguments are supplied")

    monkeypatch.setattr("app.db.sqlite.get_settings", broken_get_settings)

    db_file = tmp_path / "explicit" / "devguide.db"
    data_dir = tmp_path / "explicit"

    # Must succeed without calling get_settings
    init_db(db_path=db_file, data_dir=data_dir)

    assert db_file.is_file()

    with get_connection(db_file) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (DOCUMENTS_TABLE_NAME,),
        )
        assert cursor.fetchone() is not None


def test_init_db_calls_get_settings_when_arguments_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify init_db retrieves settings when at least one argument is omitted."""
    calls = []

    from app.core.config import Settings

    custom_settings = Settings(
        DATA_DIR=tmp_path / "settings_data",
        SQLITE_PATH=tmp_path / "settings_data" / "devguide.db",
        CHROMA_PATH=tmp_path / "settings_chroma",
    )

    def tracking_get_settings():
        calls.append("called")
        return custom_settings

    monkeypatch.setattr("app.db.sqlite.get_settings", tracking_get_settings)

    db_file = tmp_path / "partial" / "devguide.db"
    # Supply db_path but omit data_dir -> must call get_settings
    init_db(db_path=db_file)
    assert len(calls) == 1
    assert not pathlib.Path("data").exists()


def test_connection_cleanup_when_pragma_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify rollback and close are attempted even if connection setup fails before yielding."""
    mock_conn = MagicMock(spec=sqlite3.Connection)
    mock_conn.execute.side_effect = sqlite3.OperationalError("PRAGMA setup failed")

    monkeypatch.setattr("sqlite3.connect", lambda *args, **kwargs: mock_conn)

    with pytest.raises(sqlite3.OperationalError, match="PRAGMA setup failed"):
        with get_connection("dummy.db"):
            pass

    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()

