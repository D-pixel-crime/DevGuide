"""REST API route handler for end-to-end question answering (POST /api/v1/query)."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes_documents import get_app_settings
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.chroma import ChromaAdapter, ChromaError
from app.models.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    OpenAIEmbeddingAdapter,
)
from app.services.generation import (
    AnswerContextBuilder,
    GenerationAuthenticationError,
    GenerationError,
    GenerationProviderError,
    InvalidGenerationResponseError,
    OpenAIGenerationAdapter,
)
from app.services.query import QueryService
from app.services.retrieval import InvalidQueryError, RetrievalService

logger = get_logger("app.api.routes_query")

router = APIRouter(tags=["query"])


def get_query_service(
    settings: Settings = Depends(get_app_settings),
) -> QueryService:
    """Dependency provider for QueryService using application-scoped settings."""
    chroma_adapter = ChromaAdapter(chroma_path=settings.CHROMA_PATH, settings=settings)
    embedding_adapter = OpenAIEmbeddingAdapter(settings=settings)
    retrieval_service = RetrievalService(
        chroma_adapter=chroma_adapter,
        embedding_adapter=embedding_adapter,
        settings=settings,
    )
    generation_adapter = OpenAIGenerationAdapter(settings=settings)
    context_builder = AnswerContextBuilder(settings=settings)
    return QueryService(
        retrieval_service=retrieval_service,
        generation_adapter=generation_adapter,
        context_builder=context_builder,
        settings=settings,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a natural language question with evidence-grounded answer",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question string or top_k out of bounds"},
        500: {"model": ErrorResponse, "description": "Storage or generation service failure during query"},
        503: {"model": ErrorResponse, "description": "Upstream OpenAI API failure during query embedding or generation"},
    },
)
def query(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Ask a natural language question against the knowledge base.

    Performs retrieval, relevance filtering (distance <= 0.70), context formatting,
    and grounded answer generation with source citations. If no chunks pass the relevance
    threshold, returns the canonical abstention response immediately without calling the
    generation model.
    """
    try:
        result = service.query(
            question=request.question,
            top_k=request.top_k,
        )
        return QueryResponse.model_validate(result)
    except InvalidQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (
        EmbeddingAuthenticationError,
        GenerationAuthenticationError,
        EmbeddingProviderError,
        GenerationProviderError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except InvalidGenerationResponseError as exc:
        logger.error("Malformed or empty generation response: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation service returned an invalid or empty response",
        ) from exc
    except (EmbeddingError, GenerationError, ChromaError) as exc:
        logger.error("Internal service failure during query processing: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage or generation service failure during query",
        ) from exc
