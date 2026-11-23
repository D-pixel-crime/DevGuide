"""REST API route handler for semantic retrieval without generation (POST /api/v1/retrieve)."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.routes_documents import get_app_settings
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.chroma import ChromaAdapter, ChromaError
from app.models.schemas import ErrorResponse, RetrieveRequest, RetrieveResponse
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    OpenAIEmbeddingAdapter,
)
from app.services.retrieval import InvalidQueryError, RetrievalService

logger = get_logger("app.api.routes_retrieve")

router = APIRouter(tags=["retrieval"])


def get_retrieval_service(
    settings: Settings = Depends(get_app_settings),
) -> RetrievalService:
    """Dependency provider for RetrievalService using application-scoped settings."""
    chroma_adapter = ChromaAdapter(chroma_path=settings.CHROMA_PATH, settings=settings)
    embedding_adapter = OpenAIEmbeddingAdapter(settings=settings)
    return RetrievalService(
        chroma_adapter=chroma_adapter,
        embedding_adapter=embedding_adapter,
        settings=settings,
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic retrieval of relevant chunks without generation",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question string or top_k out of bounds"},
        500: {"model": ErrorResponse, "description": "Storage or ChromaDB failure during retrieval"},
        503: {"model": ErrorResponse, "description": "Upstream OpenAI API failure during query embedding"},
    },
)
def retrieve(
    request: RetrieveRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    """Search for relevant document chunks using vector cosine distance without calling the generation model.

    Evaluates nearest chunks against MAX_COSINE_DISTANCE (0.70) heuristic and returns
    canonical chunk matches preserving raw distances. Does not assign source labels [S1], [S2].
    """
    try:
        result = service.retrieve(
            question=request.question,
            top_k=request.top_k,
        )
        return RetrieveResponse.model_validate(result)
    except InvalidQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (EmbeddingProviderError, EmbeddingAuthenticationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (EmbeddingError, ChromaError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage or embedding service failure during retrieval",
        ) from exc
