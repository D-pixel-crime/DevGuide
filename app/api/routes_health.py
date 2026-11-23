"""Health and readiness probe endpoints."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["Health & Readiness"])

EVALUATED_COMPONENTS = ("sqlite", "chromadb")


def default_sqlite_check() -> str:
    """Default SQLite check before storage layer initialization in Prompt 5."""
    return "not_initialized"


def default_chromadb_check() -> str:
    """Default ChromaDB check before storage layer initialization in Prompt 11."""
    return "not_initialized"


def get_default_readiness_checks() -> dict[str, Callable[[], Any]]:
    """Return the default readiness check callables for local storage dependencies."""
    return {
        "sqlite": default_sqlite_check,
        "chromadb": default_chromadb_check,
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
        checks_map = get_default_readiness_checks()

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
