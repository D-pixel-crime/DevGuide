"""SQLite database foundation for DevGuide."""

import contextlib
import pathlib
import sqlite3
from collections.abc import Generator
from typing import Optional, Union

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger("app.db.sqlite")

DOCUMENTS_TABLE_NAME = "documents"

CREATE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    content_sha256 TEXT UNIQUE NOT NULL,
    chunk_count INTEGER NOT NULL CHECK(chunk_count >= 0),
    created_at TEXT NOT NULL
);
""".strip()


@contextlib.contextmanager
def get_connection(
    db_path: Optional[Union[pathlib.Path, str]] = None,
    timeout: float = 5.0,
) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite connection as a context manager.

    Opens the database at db_path (or configured SQLITE_PATH from Settings),
    sets row_factory to sqlite3.Row, enables foreign keys, commits on clean
    exit, rolls back on failure, and always closes the connection.
    """
    if db_path is None:
        db_path = get_settings().SQLITE_PATH

    conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def init_db(
    db_path: Optional[Union[pathlib.Path, str]] = None,
    data_dir: Optional[Union[pathlib.Path, str]] = None,
) -> None:
    """Initialize the SQLite schema and create required parent directories.

    Safe and idempotent: executes CREATE TABLE IF NOT EXISTS.
    Repeated calls do not modify or delete existing data.
    """
    if db_path is not None and data_dir is not None:
        target_path = pathlib.Path(db_path)
        target_data_dir = pathlib.Path(data_dir)
    else:
        settings = get_settings()
        target_path = (
            pathlib.Path(db_path) if db_path is not None else settings.SQLITE_PATH
        )
        target_data_dir = (
            pathlib.Path(data_dir) if data_dir is not None else settings.DATA_DIR
        )

    target_str = str(target_path)
    if target_str != ":memory:" and not target_str.startswith("file::memory:"):
        target_data_dir.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(target_path) as conn:
        conn.execute(CREATE_DOCUMENTS_TABLE_SQL)

    logger.info("SQLite database schema initialized at %s", target_path)


def check_sqlite_readiness(
    db_path: Optional[Union[pathlib.Path, str]] = None,
    timeout: float = 5.0,
) -> str:
    """Verify local SQLite database availability using SELECT 1.

    Returns:
        'ok': if SQLite connection succeeds and SELECT 1 returns 1.
        'not_initialized': if database file does not exist on disk.
        'error': if connection or query fails.
    """
    if db_path is None:
        db_path = get_settings().SQLITE_PATH

    target_path = pathlib.Path(db_path)
    target_str = str(db_path)
    is_memory = target_str == ":memory:" or target_str.startswith("file::memory:")

    if not is_memory and not target_path.exists():
        return "not_initialized"

    try:
        with get_connection(target_path, timeout=timeout) as conn:
            cursor = conn.execute("SELECT 1;")
            result = cursor.fetchone()
            if result is not None and result[0] == 1:
                return "ok"
            return "error"
    except Exception as exc:
        logger.error("SQLite readiness check failed: %s", type(exc).__name__)
        return "error"
