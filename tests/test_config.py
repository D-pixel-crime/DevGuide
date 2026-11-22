"""Tests for DevGuide configuration and settings accessor."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import app.core.config as config
from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure get_settings cache is cleared before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_configuration_values():
    """Verify exact canonical defaults and types when no overrides are present."""
    settings = Settings(_env_file=None)

    assert settings.APP_HOST == "127.0.0.1"
    assert settings.APP_PORT == 8000
    assert settings.DATA_DIR == Path("./data")
    assert settings.SQLITE_PATH == Path("./data/devguide.db")
    assert settings.CHROMA_PATH == Path("./data/chroma")
    assert settings.OPENAI_API_KEY is None or settings.OPENAI_API_KEY == ""
    assert settings.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.OPENAI_CHAT_MODEL == "gpt-6-luna"
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 150
    assert settings.EMBEDDING_BATCH_SIZE == 64
    assert settings.DEFAULT_TOP_K == 5
    assert settings.MAX_TOP_K == 10
    assert settings.MAX_COSINE_DISTANCE == 0.70
    assert settings.MAX_UPLOAD_MB == 10
    assert settings.MAX_CONTEXT_CHARS == 12000

    # Verify types
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.SQLITE_PATH, Path)
    assert isinstance(settings.CHROMA_PATH, Path)
    assert isinstance(settings.APP_PORT, int)
    assert isinstance(settings.CHUNK_SIZE, int)
    assert isinstance(settings.CHUNK_OVERLAP, int)
    assert isinstance(settings.EMBEDDING_BATCH_SIZE, int)
    assert isinstance(settings.DEFAULT_TOP_K, int)
    assert isinstance(settings.MAX_TOP_K, int)
    assert isinstance(settings.MAX_UPLOAD_MB, int)
    assert isinstance(settings.MAX_CONTEXT_CHARS, int)
    assert isinstance(settings.MAX_COSINE_DISTANCE, float)
    assert isinstance(settings.APP_HOST, str)


def test_valid_environment_variable_overrides(monkeypatch):
    """Verify that environment variables properly override defaults."""
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("DEFAULT_TOP_K", "3")
    monkeypatch.setenv("MAX_TOP_K", "8")
    monkeypatch.setenv("MAX_COSINE_DISTANCE", "0.55")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "6000")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "custom-chat")

    settings = Settings(_env_file=None)

    assert settings.APP_HOST == "0.0.0.0"
    assert settings.APP_PORT == 9000
    assert settings.CHUNK_SIZE == 500
    assert settings.CHUNK_OVERLAP == 50
    assert settings.EMBEDDING_BATCH_SIZE == 32
    assert settings.DEFAULT_TOP_K == 3
    assert settings.MAX_TOP_K == 8
    assert settings.MAX_COSINE_DISTANCE == 0.55
    assert settings.MAX_UPLOAD_MB == 25
    assert settings.MAX_CONTEXT_CHARS == 6000
    assert settings.OPENAI_EMBEDDING_MODEL == "custom-embedding"
    assert settings.OPENAI_CHAT_MODEL == "custom-chat"


def test_loading_from_temporary_dotenv(tmp_path):
    """Verify loading settings from an optional temporary .env file."""
    env_content = (
        "APP_PORT=8888\n"
        "CHUNK_SIZE=800\n"
        "CHUNK_OVERLAP=100\n"
        "DEFAULT_TOP_K=4\n"
    )
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(env_content, encoding="utf-8")

    settings = Settings(_env_file=dotenv_file)

    assert settings.APP_PORT == 8888
    assert settings.CHUNK_SIZE == 800
    assert settings.CHUNK_OVERLAP == 100
    assert settings.DEFAULT_TOP_K == 4


def test_environment_variable_precedence_over_dotenv(tmp_path, monkeypatch):
    """Verify that operating-system environment variables override values loaded from a .env file."""
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("APP_PORT=8001\n", encoding="utf-8")

    monkeypatch.setenv("APP_PORT", "9001")

    settings = Settings(_env_file=dotenv_file)

    assert settings.APP_PORT == 9001


def test_missing_dotenv_file_behavior(tmp_path):
    """Verify that a non-existent .env file does not cause errors and defaults are used."""
    non_existent = tmp_path / "non_existent.env"
    settings = Settings(_env_file=non_existent)
    assert settings.APP_PORT == 8000


def test_empty_and_missing_openai_api_key(monkeypatch):
    """Verify that empty string or missing OPENAI_API_KEY is accepted."""
    # Missing environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings_missing = Settings(_env_file=None)
    assert settings_missing.OPENAI_API_KEY is None

    # Empty string environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings_empty = Settings(_env_file=None)
    assert settings_empty.OPENAI_API_KEY in (None, "")

    # Explicit constructor arguments
    assert Settings(_env_file=None, OPENAI_API_KEY=None).OPENAI_API_KEY is None
    assert Settings(_env_file=None, OPENAI_API_KEY="").OPENAI_API_KEY in (None, "")


def test_invalid_app_host():
    """Verify APP_HOST validation rejects empty or whitespace-only strings."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_HOST="")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_HOST="   ")


def test_invalid_app_port():
    """Verify APP_PORT validation enforces 1 <= port <= 65535."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_PORT=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_PORT=-1)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_PORT=65536)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_PORT=100000)

    # Valid boundaries
    assert Settings(_env_file=None, APP_PORT=1).APP_PORT == 1
    assert Settings(_env_file=None, APP_PORT=65535).APP_PORT == 65535


def test_invalid_chunk_size_and_overlap():
    """Verify CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE."""
    # CHUNK_SIZE must be positive
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_SIZE=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_SIZE=-100)

    # CHUNK_OVERLAP must be non-negative
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_OVERLAP=-1)

    # Overlap must be strictly smaller than size
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_SIZE=500, CHUNK_OVERLAP=500)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_SIZE=500, CHUNK_OVERLAP=600)

    # Valid boundaries
    s1 = Settings(_env_file=None, CHUNK_SIZE=500, CHUNK_OVERLAP=0)
    assert s1.CHUNK_OVERLAP == 0

    s2 = Settings(_env_file=None, CHUNK_SIZE=500, CHUNK_OVERLAP=499)
    assert s2.CHUNK_OVERLAP == 499


def test_invalid_embedding_batch_size():
    """Verify EMBEDDING_BATCH_SIZE must be positive."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, EMBEDDING_BATCH_SIZE=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, EMBEDDING_BATCH_SIZE=-5)

    assert Settings(_env_file=None, EMBEDDING_BATCH_SIZE=1).EMBEDDING_BATCH_SIZE == 1


def test_invalid_default_top_k_and_max_top_k():
    """Verify MAX_TOP_K in [1, 10] and 1 <= DEFAULT_TOP_K <= MAX_TOP_K."""
    # MAX_TOP_K boundaries
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_TOP_K=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_TOP_K=11)

    # DEFAULT_TOP_K must be >= 1
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DEFAULT_TOP_K=0)

    # DEFAULT_TOP_K must not exceed MAX_TOP_K
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_TOP_K=5, DEFAULT_TOP_K=6)

    # Valid boundaries
    s = Settings(_env_file=None, MAX_TOP_K=5, DEFAULT_TOP_K=5)
    assert s.DEFAULT_TOP_K == 5
    assert s.MAX_TOP_K == 5


def test_invalid_max_cosine_distance():
    """Verify MAX_COSINE_DISTANCE must be between 0.0 and 2.0."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_COSINE_DISTANCE=-0.01)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_COSINE_DISTANCE=2.01)

    # Valid boundaries
    assert Settings(_env_file=None, MAX_COSINE_DISTANCE=0.0).MAX_COSINE_DISTANCE == 0.0
    assert Settings(_env_file=None, MAX_COSINE_DISTANCE=2.0).MAX_COSINE_DISTANCE == 2.0


def test_invalid_upload_and_context_limits():
    """Verify MAX_UPLOAD_MB and MAX_CONTEXT_CHARS must be positive."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_UPLOAD_MB=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_UPLOAD_MB=-1)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_CONTEXT_CHARS=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_CONTEXT_CHARS=-1)


def test_path_environment_overrides(monkeypatch):
    """Verify path configuration overrides and ensure no directories or files are created."""
    custom_data = Path("/nonexistent/test_data_dir")
    custom_sqlite = Path("/nonexistent/test_data_dir/test.db")
    custom_chroma = Path("/nonexistent/test_data_dir/test_chroma")

    monkeypatch.setenv("DATA_DIR", str(custom_data))
    monkeypatch.setenv("SQLITE_PATH", str(custom_sqlite))
    monkeypatch.setenv("CHROMA_PATH", str(custom_chroma))

    settings = Settings(_env_file=None)

    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.SQLITE_PATH, Path)
    assert isinstance(settings.CHROMA_PATH, Path)

    assert settings.DATA_DIR == custom_data
    assert settings.SQLITE_PATH == custom_sqlite
    assert settings.CHROMA_PATH == custom_chroma

    # Ensure no filesystem resources were created merely by initializing settings
    assert not custom_data.exists()
    assert not custom_sqlite.exists()
    assert not custom_chroma.exists()


def test_settings_accessor_caching(monkeypatch):
    """Verify get_settings returns a cached instance and cache can be cleared."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

    monkeypatch.setenv("APP_PORT", "9999")
    get_settings.cache_clear()
    s3 = get_settings()
    assert s3 is not s1
    assert s3.APP_PORT == 9999


def test_settings_accessor_override(monkeypatch):
    """Verify get_settings can be straightforwardly overridden or monkeypatched in tests."""
    mock_settings = Settings(_env_file=None, APP_PORT=12345)
    monkeypatch.setattr(config, "get_settings", lambda: mock_settings)

    active = config.get_settings()
    assert active.APP_PORT == 12345
    assert active is mock_settings
