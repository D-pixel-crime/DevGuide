"""DevGuide main application entry point and factory."""

from collections.abc import Callable
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router
from app.api.routes_health import get_default_readiness_checks
from app.api.routes_health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger("app.main")


def create_app(
    settings: Optional[Settings] = None,
    readiness_checks: Optional[dict[str, Callable[[], Any]]] = None,
) -> FastAPI:
    """Application factory for DevGuide.

    Creates and configures a FastAPI instance with standard metadata,
    routes, logging, exception handlers, and injectable dependencies.
    Does not initialize SQLite, ChromaDB, or OpenAI at import or startup.
    """
    resolved_settings = settings if settings is not None else get_settings()

    configure_logging()

    app = FastAPI(
        title="DevGuide",
        description="Local developer documentation assistant.",
        version="0.1.0",
    )

    # Store resolved settings in application state
    app.state.settings = resolved_settings

    # Injectable readiness seam (defaults to not_initialized checks)
    app.state.readiness_checks = (
        readiness_checks
        if readiness_checks is not None
        else get_default_readiness_checks()
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
