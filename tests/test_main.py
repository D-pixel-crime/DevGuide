"""Tests for the DevGuide FastAPI application shell."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import configure_logging
from app.main import create_app


def test_app_factory_defaults():
    """Verify application factory initializes with correct metadata and default settings."""
    app = create_app()

    assert isinstance(app, FastAPI)
    assert app.title == "DevGuide"
    assert app.description == "Local developer documentation assistant."
    assert app.version == "0.1.0"
    assert hasattr(app.state, "settings")
    assert isinstance(app.state.settings, Settings)
    assert app.state.settings.APP_HOST == "127.0.0.1"
    assert app.state.settings.APP_PORT == 8000


def test_app_factory_injected_settings():
    """Verify application factory accepts injected settings."""
    custom_settings = Settings(
        APP_HOST="0.0.0.0",
        APP_PORT=9090,
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50,
    )
    app = create_app(settings=custom_settings)

    assert app.state.settings is custom_settings
    assert app.state.settings.APP_HOST == "0.0.0.0"
    assert app.state.settings.APP_PORT == 9090


def test_healthz_liveness_endpoint():
    """Verify GET /healthz returns 200 OK with stable liveness status."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok"}


def test_readyz_shell_not_ready():
    """Verify GET /readyz reports not_ready with 503 for the uninitialized shell."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "not_initialized",
        "chromadb": "not_initialized",
    }
    # OpenAI must never be included in readiness checks
    assert "openai" not in data["checks"]


def test_readyz_injectable_seam_all_ready():
    """Verify GET /readyz returns 200 and ready when all checks pass."""
    app = create_app(
        readiness_checks={
            "sqlite": lambda: "ok",
            "chromadb": lambda: True,
        }
    )
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"] == {
        "sqlite": "ok",
        "chromadb": "ok",
    }


def test_readyz_injectable_seam_partial_failure():
    """Verify GET /readyz returns 503 when a check fails and sanitizes error details."""
    app = create_app()
    # Test state mutation injection
    app.state.readiness_checks = {
        "sqlite": lambda: "error: database disk image is malformed",
        "chromadb": lambda: "ok",
    }
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "error",
        "chromadb": "ok",
    }
    # Sensitive error details must not leak
    assert "malformed" not in response.text


def test_readyz_injectable_seam_exception_handling():
    """Verify GET /readyz sanitizes unhandled exceptions in checks to generic error."""
    def broken_check():
        raise RuntimeError("sensitive connection refused at /secrets/db.sqlite")

    app = create_app(
        readiness_checks={
            "sqlite": broken_check,
            "chromadb": lambda: "ok",
        }
    )
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "error",
        "chromadb": "ok",
    }
    # Ensure raw exception messages and paths are not exposed
    assert "connection refused" not in response.text
    assert "/secrets" not in response.text


def test_readyz_empty_readiness_checks():
    """Verify an empty readiness_checks dictionary returns 503 with not_initialized."""
    app = create_app(readiness_checks={})
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "not_initialized",
        "chromadb": "not_initialized",
    }


def test_readyz_missing_sqlite_check():
    """Verify missing SQLite check remains reported as not_initialized."""
    app = create_app(
        readiness_checks={
            "chromadb": lambda: "ok",
        }
    )
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "not_initialized",
        "chromadb": "ok",
    }


def test_readyz_missing_chromadb_check():
    """Verify missing ChromaDB check remains reported as not_initialized."""
    app = create_app(
        readiness_checks={
            "sqlite": lambda: "ok",
        }
    )
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"] == {
        "sqlite": "ok",
        "chromadb": "not_initialized",
    }


def test_readyz_unknown_injected_checks_not_executed():
    """Verify unknown injected checks are never executed or returned in the response."""
    mock_unknown = MagicMock(return_value="ok")
    app = create_app(
        readiness_checks={
            "sqlite": lambda: "ok",
            "chromadb": lambda: "ok",
            "redis": mock_unknown,
            "custom_service": mock_unknown,
        }
    )
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"] == {
        "sqlite": "ok",
        "chromadb": "ok",
    }
    assert "redis" not in data["checks"]
    assert "custom_service" not in data["checks"]
    mock_unknown.assert_not_called()


def test_readyz_injected_openai_check_never_executed():
    """Verify an injected OpenAI readiness check is never executed or returned."""
    mock_openai = MagicMock(return_value="ok")
    app = create_app(
        readiness_checks={
            "sqlite": lambda: "ok",
            "chromadb": lambda: "ok",
            "openai": mock_openai,
        }
    )
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "openai" not in data["checks"]
    mock_openai.assert_not_called()


def test_generic_exception_handler_unexpected_error(caplog):
    """Verify generic exception handler catches unhandled exceptions, sanitizes response and logs."""
    app = create_app()

    # Register a temporary route that triggers an unexpected exception with sensitive info
    @app.get("/test-unexpected-error")
    def fail_route():
        raise RuntimeError("Secret key leaked at /var/secrets/openai_key.txt")

    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="app.main"):
        response = client.get("/test-unexpected-error")

    assert response.status_code == 500
    data = response.json()
    assert data == {"detail": "Internal server error"}

    # Ensure sensitive details are not leaked in the HTTP response
    assert "Secret" not in response.text
    assert "/var/secrets" not in response.text

    # Ensure sensitive details and tracebacks are not leaked into the application log
    log_text = caplog.text
    assert "RuntimeError" in log_text
    assert "Secret key leaked" not in log_text
    assert "/var/secrets/openai_key.txt" not in log_text
    assert "Traceback" not in log_text


def test_generic_exception_handler_preserves_http_exceptions():
    """Verify intentional HTTPExceptions retain their status codes and details."""
    app = create_app()

    @app.get("/test-http-exception")
    def intentional_error():
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test-http-exception")

    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}


def test_generic_exception_handler_preserves_validation_errors():
    """Verify FastAPI validation errors return 422 with structured details."""
    app = create_app()

    @app.get("/test-validation/{val}")
    def validation_endpoint(val: int):
        return {"val": val}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test-validation/not-an-int")

    assert response.status_code == 422
    assert "detail" in response.json()


def test_logging_preserves_preconfigured_root_handlers():
    """Verify configure_logging does not forcibly replace existing root logger handlers."""
    root_logger = logging.getLogger()
    sentinel_handler = logging.NullHandler()
    root_logger.addHandler(sentinel_handler)
    try:
        configure_logging()
        assert sentinel_handler in root_logger.handlers
    finally:
        root_logger.removeHandler(sentinel_handler)


def test_import_and_create_app_without_openai_api_key():
    """Verify application imports and initializes without OPENAI_API_KEY."""
    with patch.dict(os.environ, {}, clear=True):
        # Explicitly remove OPENAI_API_KEY from environment
        os.environ.pop("OPENAI_API_KEY", None)

        settings = Settings()
        assert settings.OPENAI_API_KEY is None

        app = create_app(settings=settings)
        client = TestClient(app)

        health_res = client.get("/healthz")
        assert health_res.status_code == 200
        assert health_res.json() == {"status": "ok"}

        ready_res = client.get("/readyz")
        assert ready_res.status_code == 503
        assert ready_res.json()["status"] == "not_ready"


def test_no_openai_initialization_on_startup_or_health():
    """Verify no OpenAI client or API calls are initiated during app lifecycle."""
    with patch("openai.OpenAI") as mock_openai:
        app = create_app()
        client = TestClient(app)

        client.get("/healthz")
        client.get("/readyz")

        # OpenAI constructor should never be called
        mock_openai.assert_not_called()


def test_api_v1_router_registered():
    """Verify /api/v1 router is mounted on the application without business endpoints."""
    app = create_app()
    client = TestClient(app)

    # OpenAPI schema reflects registered operational endpoints
    openapi_paths = list(app.openapi()["paths"].keys())
    assert "/healthz" in openapi_paths
    assert "/readyz" in openapi_paths

    # Confirm no business endpoints have been prematurely registered
    assert "/api/v1/documents" not in openapi_paths
    assert "/api/v1/retrieve" not in openapi_paths
    assert "/api/v1/query" not in openapi_paths

    # Confirm unmapped /api/v1 routes return 404
    response = client.get("/api/v1/documents")
    assert response.status_code == 404


def test_no_database_files_created_by_shell(tmp_path):
    """Verify running the shell and health/readiness probes does not create database files."""
    db_file = tmp_path / "devguide.db"
    chroma_dir = tmp_path / "chroma"

    settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_file,
        CHROMA_PATH=chroma_dir,
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    client.get("/healthz")
    client.get("/readyz")

    # Storage paths must not be created by shell endpoints
    assert not db_file.exists()
    assert not chroma_dir.exists()
