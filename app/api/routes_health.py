"""Health and readiness probe endpoints."""

import pathlib
from collections.abc import Callable
from typing import Any, Optional, Union

from fastapi import APIRouter, Request, Response, status

from app.core.config import Settings, get_settings
from app.db.chroma import check_chroma_readiness
from app.db.sqlite import check_sqlite_readiness

router = APIRouter(tags=["Health & Readiness"])

EVALUATED_COMPONENTS = ("sqlite", "chromadb")


def default_chromadb_check(
    chroma_path: Optional[Union[pathlib.Path, str]] = None,
) -> str:
    """Check ChromaDB readiness without creating storage paths as a side effect."""
    return check_chroma_readiness(chroma_path)


def get_default_readiness_checks(
    settings: Optional[Settings] = None,
) -> dict[str, Callable[[], Any]]:
    """Return the default readiness check callables for local storage dependencies."""
    target_settings = settings if settings is not None else get_settings()
    return {
        "sqlite": lambda: check_sqlite_readiness(target_settings.SQLITE_PATH),
        "chromadb": lambda: check_chroma_readiness(target_settings.CHROMA_PATH),
    }


@router.get("/healthz", summary="Liveness probe")
def healthz() -> dict[str, str]:
    """Process liveness probe confirming the application process is running.

    This check is completely decoupled from SQLite, ChromaDB, OpenAI,
    or any external services.
    """
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readyz(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe verifying local storage dependencies.

    Only checks SQLite and ChromaDB availability. OpenAI is never checked
    and a missing or empty OPENAI_API_KEY does not affect readiness.

    Returns HTTP 200 with status 'ready' when all checks succeed.
    Returns HTTP 503 with status 'not_ready' if any check fails or is uninitialized.
    """
    checks_map: dict[str, Callable[[], Any]] | None = getattr(
        request.app.state, "readiness_checks", None
    )
    if checks_map is None:
        app_settings = getattr(request.app.state, "settings", None)
        checks_map = get_default_readiness_checks(app_settings)

    results: dict[str, str] = {}
    all_ready = True

    for component in EVALUATED_COMPONENTS:
        check_fn = checks_map.get(component)
        if check_fn is None:
            val = "not_initialized"
        else:
            try:
                check_result = check_fn()
                if check_result is True or check_result == "ok":
                    val = "ok"
                elif check_result == "not_initialized":
                    val = "not_initialized"
                else:
                    val = "error"
            except Exception:
                val = "error"

        results[component] = val
        if val != "ok":
            all_ready = False

    if all_ready:
        return {
            "status": "ready",
            "checks": results,
        }

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "not_ready",
        "checks": results,
    }
