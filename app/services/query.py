"""RAG query orchestration service coordinating retrieval, relevance decision, and generation."""

from dataclasses import dataclass
import re
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import QuerySource
from app.services.generation import (
    CANONICAL_ABSTENTION_ANSWER,
    AnswerContextBuilder,
    BuiltContext,
    InvalidGenerationResponseError,
    OpenAIGenerationAdapter,
)
from app.services.retrieval import (
    RetrievalResult,
    RetrievalService,
)

logger = get_logger("app.services.query")


class QueryServiceError(Exception):
    """Base exception for query service errors."""


def is_citation_like_token(token_text: str) -> bool:
    """Determine whether bracketed content appears to be a citation reference.

    Matches tokens that resemble citations, including:
    - Standard or malformed letter-number combinations: 'S1', 'C1', 's1', 'S 2', 'S-1', 'S_1'
    - Word prefixes with numbers: 'Source 1', 'source1', 'doc1', 'ref 2'
    - Pure numeric citations: '1', '2'
    - Single letter marker candidates: 'S', 's', 'C'
    - Comma-joined markers: 'S1, S2'

    Does not match ordinary prose in brackets such as '[production]', '[note]', '[optional]',
    or '[string, required]'.
    """
    t = token_text.strip()
    return bool(
        re.match(
            r"^(?:[A-Za-z]{1,6}[\s\-_]*\d+|\d+|[A-Za-z]|S\d+(?:\s*,\s*S\d+)+)$",
            t,
            re.IGNORECASE,
        )
    )


@dataclass(frozen=True)
class QueryResult:
    """Aggregated query result containing generated answer or abstention and citations.

    Attributes:
        question: The user query string.
        answer: The generated answer prose or canonical abstention response.
        abstained: True if no relevant context passed threshold (generation skipped), False otherwise.
        retrieved_count: Number of relevant sources included in context (0 on abstention).
        sources: Structured source objects mapping citation labels to chunk metadata.
    """

    question: str
    answer: str
    abstained: bool
    retrieved_count: int
    sources: list[QuerySource]


class QueryService:
    """End-to-end RAG question answering orchestration service.

    Coordinates:
    1. Query validation and retrieval via RetrievalService (embedding, vector search, distance <= 0.70 filtering).
    2. Relevance decision: if no chunks pass threshold, return fixed canonical abstention without calling LLM.
    3. Bounded context assembly adhering to MAX_CONTEXT_CHARS ceiling with deterministic [S1], [S2] source labels.
    4. Grounded answer generation via OpenAIGenerationAdapter (OpenAI Responses API with gpt-6-luna).
    5. Structured response pairing generated answer with independent application-owned QuerySource metadata.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_adapter: OpenAIGenerationAdapter,
        context_builder: Optional[AnswerContextBuilder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize query service with retrieval, generation, and context components.

        Args:
            retrieval_service: Configured RetrievalService instance.
            generation_adapter: Configured OpenAIGenerationAdapter instance.
            context_builder: Optional AnswerContextBuilder instance.
            settings: Optional application settings.
        """
        self._retrieval_service = retrieval_service
        self._generation_adapter = generation_adapter
        self._context_builder = (
            context_builder
            if context_builder is not None
            else AnswerContextBuilder(settings=settings)
        )
        self._settings = settings if settings is not None else get_settings()

    @property
    def retrieval_service(self) -> RetrievalService:
        """Underlying retrieval service."""
        return self._retrieval_service

    @property
    def generation_adapter(self) -> OpenAIGenerationAdapter:
        """Underlying generation adapter."""
        return self._generation_adapter

    @property
    def context_builder(self) -> AnswerContextBuilder:
        """Underlying answer context builder."""
        return self._context_builder

    @property
    def settings(self) -> Settings:
        """Active application settings."""
        return self._settings

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """Execute end-to-end question answering pipeline.

        Args:
            question: Non-empty question string (max 1000 characters).
            top_k: Optional nearest neighbors limit (1..MAX_TOP_K). Defaults to DEFAULT_TOP_K.

        Returns:
            QueryResult with generated answer and source citations, or canonical abstention.

        Raises:
            TypeError: If question is not a string.
            InvalidQueryError: If question is empty or top_k is out of range [1, MAX_TOP_K].
            EmbeddingError: If embedding generation fails.
            GenerationError: If answer generation fails.
            ChromaError: If ChromaDB query fails.
        """
        # 1. Retrieve relevant chunks passing MAX_COSINE_DISTANCE heuristic
        retrieval_result: RetrievalResult = self._retrieval_service.retrieve(
            question=question,
            top_k=top_k,
        )

        # 2. Relevance decision & abstention evaluation
        # If no relevant chunks remain, return exact fixed abstention immediately
        # and DO NOT call generation adapter.
        if not retrieval_result.has_relevant_context or not retrieval_result.matches:
            logger.info(
                "No relevant context passed distance threshold (<= %.2f). Abstaining without calling generation model.",
                self._settings.MAX_COSINE_DISTANCE,
            )
            return QueryResult(
                question=retrieval_result.question,
                answer=CANONICAL_ABSTENTION_ANSWER,
                abstained=True,
                retrieved_count=0,
                sources=[],
            )

        # 3. Assemble bounded context adhering to MAX_CONTEXT_CHARS ceiling
        built_context: BuiltContext = self._context_builder.build(
            question=retrieval_result.question,
            matches=retrieval_result.matches,
            max_context_chars=self._settings.MAX_CONTEXT_CHARS,
        )

        # If budget constraint omitted all candidate matches, abstain safely without calling model
        if not built_context.sources or built_context.included_count == 0:
            logger.warning(
                "All candidate matches omitted due to context budget limits (%d chars). Abstaining.",
                self._settings.MAX_CONTEXT_CHARS,
            )
            return QueryResult(
                question=retrieval_result.question,
                answer=CANONICAL_ABSTENTION_ANSWER,
                abstained=True,
                retrieved_count=0,
                sources=[],
            )

        # 4. Generate grounded answer strictly from bounded evidence
        logger.info(
            "Calling generation model for question with %d bounded sources (%d context chars)",
            built_context.included_count,
            built_context.total_chars,
        )
        generation_result = self._generation_adapter.generate(
            question=retrieval_result.question,
            context=built_context,
        )

        # 5. Validate generated source citations
        raw_answer = generation_result.answer.strip()
        if raw_answer == CANONICAL_ABSTENTION_ANSWER:
            logger.info("Generation model returned canonical abstention answer.")
            return QueryResult(
                question=retrieval_result.question,
                answer=CANONICAL_ABSTENTION_ANSWER,
                abstained=True,
                retrieved_count=0,
                sources=[],
            )

        valid_labels = {source.source_label for source in built_context.sources}

        # Find all bracketed tokens (including those followed by markdown link syntax)
        bracket_tokens = re.findall(r"\[([^\]]+)\]", raw_answer)
        citation_like_tokens = [t.strip() for t in bracket_tokens if is_citation_like_token(t)]

        if not citation_like_tokens:
            raise InvalidGenerationResponseError(
                "Generated answer contains no source citations"
            )

        valid_citation_pattern = re.compile(r"^S[1-9]\d*$")
        for token in citation_like_tokens:
            if not valid_citation_pattern.match(token):
                raise InvalidGenerationResponseError(
                    f"Generated answer cited malformed citation token '[{token}]'"
                )
            if token not in valid_labels:
                raise InvalidGenerationResponseError(
                    f"Generated answer cited invalid or omitted source label '[{token}]'"
                )


        # 6. Return answer with independent structured sources from application retrieval objects
        return QueryResult(
            question=retrieval_result.question,
            answer=generation_result.answer,
            abstained=False,
            retrieved_count=built_context.included_count,
            sources=built_context.sources,
        )

