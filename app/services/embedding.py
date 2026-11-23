"""OpenAI embedding adapter with batched processing and strict validation."""

import math
from collections.abc import Sequence
from typing import Any, Optional

import openai

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger("app.services.embedding")


class EmbeddingError(Exception):
    """Base exception for all embedding adapter errors."""


class MissingApiKeyError(EmbeddingError, ValueError):
    """Raised when OPENAI_API_KEY is missing or empty upon invoking an embedding operation."""


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when the upstream provider rejects authentication (e.g. invalid API key)."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the upstream provider fails (rate limits, timeouts, server errors)."""


class InvalidEmbeddingResponseError(EmbeddingError, ValueError):
    """Raised when provider output is malformed, missing vectors, or contains non-finite numbers."""


class OpenAIEmbeddingAdapter:
    """Injectable adapter isolating the official OpenAI Python SDK for embeddings.

    Implements batched embedding generation, strict input/output ordering preservation,
    finite numeric vector validation, and secure error handling without logging sensitive data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        client: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize adapter without requiring OPENAI_API_KEY at startup.

        Args:
            api_key: Optional explicit OpenAI API key.
            model: Optional model name (defaults to Settings.OPENAI_EMBEDDING_MODEL).
            batch_size: Optional batch size (defaults to Settings.EMBEDDING_BATCH_SIZE).
            client: Optional injectable client double for testing.
            settings: Optional Settings instance.
        """
        self._explicit_api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._client = client
        self._settings = settings

    @property
    def model(self) -> str:
        """Resolved embedding model name."""
        if self._model is not None:
            return self._model
        settings = self._settings if self._settings is not None else get_settings()
        return settings.OPENAI_EMBEDDING_MODEL

    @property
    def batch_size(self) -> int:
        """Resolved embedding batch size."""
        if self._batch_size is not None:
            return self._batch_size
        settings = self._settings if self._settings is not None else get_settings()
        return settings.EMBEDDING_BATCH_SIZE

    def _get_client(self) -> Any:
        """Resolve or lazily initialize the OpenAI client.

        Raises:
            MissingApiKeyError: If no client was injected and OPENAI_API_KEY is missing or empty.
        """
        if self._client is not None:
            return self._client

        key = self._explicit_api_key
        if not key:
            settings = self._settings if self._settings is not None else get_settings()
            key = settings.OPENAI_API_KEY

        if not key or not key.strip():
            raise MissingApiKeyError(
                "OPENAI_API_KEY is not set or empty. A valid API key is required to perform embedding operations."
            )

        self._client = openai.OpenAI(api_key=key.strip())
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a sequence of texts using batched provider calls.

        Preserves exact input/output order, validates finite float vectors,
        and guarantees one vector per input item.

        Args:
            texts: Sequence of non-empty text strings to embed.

        Returns:
            List of float vectors corresponding 1:1 to the input texts.

        Raises:
            MissingApiKeyError: If OPENAI_API_KEY is missing when an embedding operation is invoked.
            EmbeddingAuthenticationError: If provider rejects authentication.
            EmbeddingProviderError: If upstream API request fails.
            InvalidEmbeddingResponseError: If response is malformed, has count mismatch, or non-finite values.
            TypeError: If input items are not strings.
            ValueError: If input items are empty or whitespace-only.
        """
        if texts is None:
            raise ValueError("texts sequence must not be None")

        if len(texts) == 0:
            return []

        # Validate input items before network calls
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Item at index {idx} must be a string, got {type(text).__name__}"
                )
            if not text.strip():
                raise ValueError(
                    f"Item at index {idx} must not be empty or whitespace-only"
                )

        batch_size = self.batch_size
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        client = self._get_client()
        model_name = self.model
        total_texts = len(texts)
        num_batches = math.ceil(total_texts / batch_size)

        logger.info(
            "Generating embeddings for %d texts in %d batches (model=%s, batch_size=%d)",
            total_texts,
            num_batches,
            model_name,
            batch_size,
        )

        all_vectors: list[list[float]] = []

        for start_idx in range(0, total_texts, batch_size):
            batch_texts = list(texts[start_idx : start_idx + batch_size])
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=model_name,
                )
            except openai.AuthenticationError as exc:
                logger.error("OpenAI authentication error: %s", type(exc).__name__)
                raise EmbeddingAuthenticationError(
                    "OpenAI authentication failed: invalid or unauthorized API key"
                ) from exc
            except openai.RateLimitError as exc:
                logger.error("OpenAI rate limit error: %s", type(exc).__name__)
                raise EmbeddingProviderError(
                    "OpenAI rate limit exceeded during embedding generation"
                ) from exc
            except openai.APITimeoutError as exc:
                logger.error("OpenAI timeout error: %s", type(exc).__name__)
                raise EmbeddingProviderError(
                    "OpenAI API request timed out during embedding generation"
                ) from exc
            except openai.APIConnectionError as exc:
                logger.error("OpenAI connection error: %s", type(exc).__name__)
                raise EmbeddingProviderError(
                    "Failed to connect to OpenAI API during embedding generation"
                ) from exc
            except openai.APIStatusError as exc:
                logger.error("OpenAI status error: %s (status=%s)", type(exc).__name__, getattr(exc, "status_code", "unknown"))
                raise EmbeddingProviderError(
                    f"OpenAI API returned status code {getattr(exc, 'status_code', 'error')} during embedding generation"
                ) from exc
            except openai.OpenAIError as exc:
                logger.error("OpenAI provider error: %s", type(exc).__name__)
                raise EmbeddingProviderError(
                    f"OpenAI API error occurred: {type(exc).__name__}"
                ) from exc
            except Exception as exc:
                logger.error("Unexpected error in embedding client: %s", type(exc).__name__)
                raise EmbeddingProviderError(
                    f"Unexpected error during embedding generation: {type(exc).__name__}"
                ) from exc

            # Validate provider response structure
            if response is None or not hasattr(response, "data") or response.data is None:
                raise InvalidEmbeddingResponseError(
                    "OpenAI embedding response is missing the 'data' field or is None"
                )

            data_items = list(response.data)
            if len(data_items) != len(batch_texts):
                raise InvalidEmbeddingResponseError(
                    f"Embedding batch count mismatch: expected {len(batch_texts)} items, received {len(data_items)}"
                )

            # Strictly validate returned item indexes
            batch_indices: list[int] = []
            for item_pos, item in enumerate(data_items):
                # 1. Require every returned embedding item to have an index
                has_index = False
                raw_index = None
                if hasattr(item, "index"):
                    raw_index = item.index
                    has_index = True
                elif isinstance(item, dict) and "index" in item:
                    raw_index = item["index"]
                    has_index = True

                if not has_index or raw_index is None:
                    raise InvalidEmbeddingResponseError(
                        f"Embedding response item at position {item_pos} is missing required 'index' field"
                    )

                # 2. Require each index to be an integer, excluding booleans
                if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                    raise InvalidEmbeddingResponseError(
                        f"Embedding response item at position {item_pos} has non-integer index: "
                        f"expected int, got {type(raw_index).__name__} ({raw_index!r})"
                    )

                # Reject negative or out-of-range index
                if raw_index < 0 or raw_index >= len(batch_texts):
                    raise InvalidEmbeddingResponseError(
                        f"Embedding response item at position {item_pos} has out-of-range index {raw_index}: "
                        f"expected range [0, {len(batch_texts) - 1}]"
                    )

                batch_indices.append(raw_index)

            # 3. Require the indexes to form exactly the sequence 0..N-1 for the current batch
            # Reject duplicate indexes
            if len(set(batch_indices)) != len(batch_texts):
                raise InvalidEmbeddingResponseError(
                    f"Duplicate embedding response indexes detected in batch: {batch_indices}"
                )

            # Reject sequence mismatch
            if set(batch_indices) != set(range(len(batch_texts))):
                raise InvalidEmbeddingResponseError(
                    f"Embedding response indexes do not form exact sequence 0..{len(batch_texts) - 1}: {batch_indices}"
                )

            # 5. Sort the response using these validated indexes before extracting vectors
            indexed_items = sorted(
                zip(batch_indices, data_items),
                key=lambda pair: pair[0],
            )

            batch_vectors: list[list[float]] = []
            for expected_idx, (item_idx, item) in enumerate(indexed_items):
                raw_embedding = (
                    item.embedding
                    if hasattr(item, "embedding")
                    else (item.get("embedding") if isinstance(item, dict) else None)
                )

                if raw_embedding is None:
                    raise InvalidEmbeddingResponseError(
                        f"Missing embedding vector at item index {item_idx}"
                    )

                if not isinstance(raw_embedding, (list, tuple)):
                    raise InvalidEmbeddingResponseError(
                        f"Embedding vector at item index {item_idx} must be a list or tuple, got {type(raw_embedding).__name__}"
                    )

                if len(raw_embedding) == 0:
                    raise InvalidEmbeddingResponseError(
                        f"Embedding vector at item index {item_idx} is empty"
                    )

                vector: list[float] = []
                for elem_idx, val in enumerate(raw_embedding):
                    if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                        raise InvalidEmbeddingResponseError(
                            f"Embedding vector at item {item_idx}, element {elem_idx} contains non-finite or non-numeric value"
                        )
                    vector.append(float(val))

                batch_vectors.append(vector)

            all_vectors.extend(batch_vectors)

        if len(all_vectors) != total_texts:
            raise InvalidEmbeddingResponseError(
                f"Total embedding count mismatch: expected {total_texts}, produced {len(all_vectors)}"
            )

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Generate a single embedding vector for a query string.

        Args:
            query: Non-empty query string.

        Returns:
            Float vector of embedding values.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        vectors = self.embed_texts([query])
        return vectors[0]
