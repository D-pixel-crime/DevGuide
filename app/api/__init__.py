"""API routing and endpoint handlers."""

from fastapi import APIRouter

from app.api.routes_documents import router as documents_router
from app.api.routes_retrieve import router as retrieve_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents_router)
api_router.include_router(retrieve_router)

__all__ = ["api_router"]
