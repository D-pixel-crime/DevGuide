"""DevGuide main application entry point and factory."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router
from app.api.routes_health import default_chromadb_check
from app.api.routes_health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.chroma import check_chroma_readiness
from app.db.sqlite import check_sqlite_readiness, init_db

logger = get_logger("app.main")


@asynccontextmanager
async def default_lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to safely initialize SQLite schema on startup."""
    app_settings: Settings = (
        getattr(app_instance.state, "settings", None) or get_settings()
    )
    try:
        init_db(app_settings.SQLITE_PATH, app_settings.DATA_DIR)
        app_instance.state.sqlite_initialized = True
        logger.info("SQLite database initialized at %s", app_settings.SQLITE_PATH)
    except Exception as exc:
        logger.error(
            "Failed to initialize SQLite database during startup: %s",
            type(exc).__name__,
        )
        app_instance.state.sqlite_initialized = False
        app_instance.state.sqlite_init_error = exc
    yield


def create_app(
    settings: Optional[Settings] = None,
    readiness_checks: Optional[dict[str, Callable[[], Any]]] = None,
    lifespan_handler: Optional[Any] = None,
) -> FastAPI:
    """Application factory for DevGuide.

    Creates and configures a FastAPI instance with standard metadata,
    routes, logging, exception handlers, and injectable dependencies.
    Does not initialize SQLite, ChromaDB, or OpenAI at import time.
    """
    resolved_settings = settings if settings is not None else get_settings()

    configure_logging()

    effective_lifespan = (
        lifespan_handler if lifespan_handler is not None else default_lifespan
    )

    app = FastAPI(
        title="DevGuide",
        description="Local developer documentation assistant.",
        version="0.1.0",
        lifespan=effective_lifespan,
    )

    # Store resolved settings in application state
    app.state.settings = resolved_settings

    # Injectable readiness seam (defaults to SQLite actual check + ChromaDB check)
    def sqlite_readiness_wrapper() -> str:
        if getattr(app.state, "sqlite_init_error", None) is not None:
            return "error"
        return check_sqlite_readiness(resolved_settings.SQLITE_PATH)

    def chroma_readiness_wrapper() -> str:
        return check_chroma_readiness(resolved_settings.CHROMA_PATH)

    app.state.readiness_checks = (
        readiness_checks
        if readiness_checks is not None
        else {
            "sqlite": sqlite_readiness_wrapper,
            "chromadb": chroma_readiness_wrapper,
        }
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(api_router)

    # Global generic exception handler for unexpected application errors
    @app.exception_handler(Exception)
    def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors without leaking internals or swallowing HTTPExceptions."""
        if isinstance(exc, StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=getattr(exc, "headers", None),
            )

        logger.error(
            "Unexpected error during request processing: %s",
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


# Export module-level app instance for Uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
