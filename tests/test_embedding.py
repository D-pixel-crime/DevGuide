"""Comprehensive tests for the OpenAI embedding adapter."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import openai
import pytest

from app.core.config import Settings
from app.services.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingProviderError,
    InvalidEmbeddingResponseError,
    MissingApiKeyError,
    OpenAIEmbeddingAdapter,
)


@dataclass
class FakeEmbeddingItem:
    """Fake embedding item returned by FakeEmbeddingData."""

    embedding: list[float]
    index: int


@dataclass
class FakeEmbeddingResponse:
    """Fake response mimicking openai.types.CreateEmbeddingResponse."""

    data: list[FakeEmbeddingItem]


class FakeOpenAIEmbeddings:
    """Fake embeddings resource mimicking openai.resources.Embeddings."""

    def __init__(
        self,
        dimension: int = 1536,
        scramble_order: bool = False,
        custom_response: Optional[Any] = None,
        exception_to_raise: Optional[Exception] = None,
    ):
        self.dimension = dimension
        self.scramble_order = scramble_order
        self.custom_response = custom_response
        self.exception_to_raise = exception_to_raise
        self.calls: list[dict[str, Any]] = []

    def create(self, *, input: list[str], model: str, **kwargs: Any) -> Any:
        self.calls.append({"input": list(input), "model": model, "kwargs": kwargs})

        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        if self.custom_response is not None:
            return self.custom_response

        items = []
        for idx, text in enumerate(input):
            # Deterministic vector based on index and dimension
            vector = [float(idx + 1) * 0.01 + float(j) * 0.0001 for j in range(self.dimension)]
            items.append(FakeEmbeddingItem(embedding=vector, index=idx))

        if self.scramble_order:
            # Reverse order of items to test index re-sorting
            items = list(reversed(items))

        return FakeEmbeddingResponse(data=items)


class FakeOpenAIClient:
    """Fake client isolating the official OpenAI Python SDK."""

    def __init__(self, embeddings: Optional[FakeOpenAIEmbeddings] = None):
        self.embeddings = embeddings if embeddings is not None else FakeOpenAIEmbeddings()


def test_empty_input_behavior():
    """Verify embed_texts([]) returns [] immediately without calling provider or checking key."""
    fake_client = FakeOpenAIClient()
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    result = adapter.embed_texts([])

    assert result == []
    assert len(fake_client.embeddings.calls) == 0


def test_empty_input_without_api_key():
    """Verify embed_texts([]) succeeds even when OPENAI_API_KEY is not set."""
    settings = Settings(OPENAI_API_KEY=None)
    adapter = OpenAIEmbeddingAdapter(settings=settings)

    result = adapter.embed_texts([])

    assert result == []


def test_missing_api_key_at_invocation():
    """Verify missing OPENAI_API_KEY raises MissingApiKeyError when embedding operation is invoked."""
    settings = Settings(OPENAI_API_KEY=None)
    adapter = OpenAIEmbeddingAdapter(settings=settings)

    with pytest.raises(MissingApiKeyError) as exc_info:
        adapter.embed_texts(["Test sentence."])

    assert "OPENAI_API_KEY is not set or empty" in str(exc_info.value)


def test_whitespace_api_key_at_invocation():
    """Verify whitespace-only OPENAI_API_KEY raises MissingApiKeyError upon invocation."""
    adapter = OpenAIEmbeddingAdapter(api_key="   ")

    with pytest.raises(MissingApiKeyError):
        adapter.embed_texts(["Test sentence."])


def test_injected_client_bypasses_api_key_requirement():
    """Verify an injected client does not require an API key to be set."""
    fake_client = FakeOpenAIClient()
    adapter = OpenAIEmbeddingAdapter(api_key=None, client=fake_client)

    result = adapter.embed_texts(["Hello world"])

    assert len(result) == 1
    assert len(result[0]) == 1536
    assert len(fake_client.embeddings.calls) == 1


def test_batched_embedding_execution():
    """Verify texts are split into batches matching EMBEDDING_BATCH_SIZE."""
    fake_embeddings = FakeOpenAIEmbeddings(dimension=8)
    fake_client = FakeOpenAIClient(fake_embeddings)
    adapter = OpenAIEmbeddingAdapter(
        batch_size=5,
        model="text-embedding-3-small",
        client=fake_client,
    )

    texts = [f"Text chunk number {i}" for i in range(13)]
    result = adapter.embed_texts(texts)

    assert len(result) == 13
    # 13 items with batch_size=5 -> 3 batches (5, 5, 3)
    assert len(fake_embeddings.calls) == 3
    assert len(fake_embeddings.calls[0]["input"]) == 5
    assert len(fake_embeddings.calls[1]["input"]) == 5
    assert len(fake_embeddings.calls[2]["input"]) == 3
    assert fake_embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_exact_input_output_order_preservation():
    """Verify input/output order is strictly preserved even if provider returns scrambled items."""
    fake_embeddings = FakeOpenAIEmbeddings(dimension=4, scramble_order=True)
    fake_client = FakeOpenAIClient(fake_embeddings)
    adapter = OpenAIEmbeddingAdapter(batch_size=10, client=fake_client)

    texts = ["First", "Second", "Third", "Fourth"]
    result = adapter.embed_texts(texts)

    assert len(result) == 4
    # With index sorting, index 0 vector starts with 0.01, index 1 with 0.02, etc.
    assert pytest.approx(result[0][0], rel=1e-3) == 0.01
    assert pytest.approx(result[1][0], rel=1e-3) == 0.02
    assert pytest.approx(result[2][0], rel=1e-3) == 0.03
    assert pytest.approx(result[3][0], rel=1e-3) == 0.04


def test_one_finite_numeric_vector_per_input():
    """Verify exactly one finite numeric vector per input is returned."""
    fake_embeddings = FakeOpenAIEmbeddings(dimension=16)
    fake_client = FakeOpenAIClient(fake_embeddings)
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    texts = ["Alpha", "Beta", "Gamma"]
    vectors = adapter.embed_texts(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) == 16
        for elem in vec:
            assert isinstance(elem, float)
            assert not isinstance(elem, bool)


def test_malformed_provider_output_nan():
    """Verify NaN in provider vector raises InvalidEmbeddingResponseError."""
    bad_data = [FakeEmbeddingItem(embedding=[0.1, float("nan"), 0.3], index=0)]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["Sample text"])

    assert "non-finite or non-numeric" in str(exc_info.value)


def test_malformed_provider_output_infinity():
    """Verify Infinity in provider vector raises InvalidEmbeddingResponseError."""
    bad_data = [FakeEmbeddingItem(embedding=[0.1, float("inf"), 0.3], index=0)]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["Sample text"])

    assert "non-finite or non-numeric" in str(exc_info.value)


def test_malformed_provider_output_non_numeric():
    """Verify string or bool in provider vector raises InvalidEmbeddingResponseError."""
    bad_data = [FakeEmbeddingItem(embedding=[0.1, "not_a_number", 0.3], index=0)]  # type: ignore
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["Sample text"])

    assert "non-finite or non-numeric" in str(exc_info.value)


def test_malformed_provider_output_empty_vector():
    """Verify empty vector in provider output raises InvalidEmbeddingResponseError."""
    bad_data = [FakeEmbeddingItem(embedding=[], index=0)]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["Sample text"])

    assert "empty" in str(exc_info.value)


def test_malformed_provider_output_count_mismatch():
    """Verify count mismatch between inputs and provider response items raises InvalidEmbeddingResponseError."""
    # Input has 2 texts, but provider returns only 1 item
    bad_data = [FakeEmbeddingItem(embedding=[0.1, 0.2], index=0)]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["First", "Second"])

    assert "Embedding batch count mismatch" in str(exc_info.value)


def test_malformed_provider_output_missing_data():
    """Verify missing data field raises InvalidEmbeddingResponseError."""
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=object())
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["Text"])

    assert "missing the 'data' field" in str(exc_info.value)


def test_malformed_provider_output_index_mismatch():
    """Verify inconsistent item indices raise InvalidEmbeddingResponseError."""
    bad_data = [FakeEmbeddingItem(embedding=[0.1, 0.2], index=99)]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc_info:
        adapter.embed_texts(["Single text"])

    assert "index" in str(exc_info.value).lower()


def test_embedding_response_missing_index():
    """Verify embedding item missing index attribute or having index=None raises InvalidEmbeddingResponseError."""
    class ItemWithoutIndex:
        embedding = [0.1, 0.2]

    fake_client1 = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=[ItemWithoutIndex()]))  # type: ignore
    )
    adapter1 = OpenAIEmbeddingAdapter(client=fake_client1)
    with pytest.raises(InvalidEmbeddingResponseError) as exc1:
        adapter1.embed_texts(["Text"])
    assert "missing required 'index'" in str(exc1.value)

    bad_item_none = FakeEmbeddingItem(embedding=[0.1, 0.2], index=None)  # type: ignore
    fake_client2 = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=[bad_item_none]))
    )
    adapter2 = OpenAIEmbeddingAdapter(client=fake_client2)
    with pytest.raises(InvalidEmbeddingResponseError) as exc2:
        adapter2.embed_texts(["Text"])
    assert "missing required 'index'" in str(exc2.value)


def test_embedding_response_duplicate_indexes():
    """Verify duplicate indexes in provider response raise InvalidEmbeddingResponseError."""
    bad_data = [
        FakeEmbeddingItem(embedding=[0.1, 0.2], index=0),
        FakeEmbeddingItem(embedding=[0.3, 0.4], index=0),
    ]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc:
        adapter.embed_texts(["First text", "Second text"])
    assert "duplicate" in str(exc.value).lower()


def test_embedding_response_negative_index():
    """Verify negative index raises InvalidEmbeddingResponseError."""
    bad_data = [FakeEmbeddingItem(embedding=[0.1, 0.2], index=-1)]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc:
        adapter.embed_texts(["Text"])
    assert "out-of-range index -1" in str(exc.value)


def test_embedding_response_out_of_range_index():
    """Verify out-of-range index raises InvalidEmbeddingResponseError."""
    bad_data = [
        FakeEmbeddingItem(embedding=[0.1, 0.2], index=0),
        FakeEmbeddingItem(embedding=[0.3, 0.4], index=5),
    ]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=bad_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(InvalidEmbeddingResponseError) as exc:
        adapter.embed_texts(["First", "Second"])
    assert "out-of-range index 5" in str(exc.value)


def test_embedding_response_non_integer_index():
    """Verify float, string, and boolean indexes raise InvalidEmbeddingResponseError."""
    fake_client_float = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1, 0.2], index=0.0)]))  # type: ignore
    )
    adapter_float = OpenAIEmbeddingAdapter(client=fake_client_float)
    with pytest.raises(InvalidEmbeddingResponseError) as exc1:
        adapter_float.embed_texts(["Text"])
    assert "non-integer index" in str(exc1.value)

    fake_client_str = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1, 0.2], index="0")]))  # type: ignore
    )
    adapter_str = OpenAIEmbeddingAdapter(client=fake_client_str)
    with pytest.raises(InvalidEmbeddingResponseError) as exc2:
        adapter_str.embed_texts(["Text"])
    assert "non-integer index" in str(exc2.value)

    fake_client_bool = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1, 0.2], index=True)]))  # type: ignore
    )
    adapter_bool = OpenAIEmbeddingAdapter(client=fake_client_bool)
    with pytest.raises(InvalidEmbeddingResponseError) as exc3:
        adapter_bool.embed_texts(["Text"])
    assert "non-integer index" in str(exc3.value)


def test_embedding_response_scrambled_response_order():
    """Verify valid response items returned out of order are sorted correctly to match input order."""
    scrambled_data = [
        FakeEmbeddingItem(embedding=[0.3, 0.3], index=2),
        FakeEmbeddingItem(embedding=[0.1, 0.1], index=0),
        FakeEmbeddingItem(embedding=[0.2, 0.2], index=1),
    ]
    fake_client = FakeOpenAIClient(
        FakeOpenAIEmbeddings(custom_response=FakeEmbeddingResponse(data=scrambled_data))
    )
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    vectors = adapter.embed_texts(["T0", "T1", "T2"])

    assert len(vectors) == 3
    assert vectors[0] == [0.1, 0.1]
    assert vectors[1] == [0.2, 0.2]
    assert vectors[2] == [0.3, 0.3]


def test_input_validation_empty_string_rejected():
    """Verify empty or whitespace-only input string is rejected."""
    fake_client = FakeOpenAIClient()
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(ValueError) as exc_info:
        adapter.embed_texts(["Valid", "   ", "Also valid"])

    assert "empty or whitespace-only" in str(exc_info.value)


def test_input_validation_non_string_rejected():
    """Verify non-string input item raises TypeError."""
    fake_client = FakeOpenAIClient()
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(TypeError) as exc_info:
        adapter.embed_texts(["Valid", 12345])  # type: ignore

    assert "must be a string" in str(exc_info.value)


def test_provider_exception_authentication_error():
    """Verify openai.AuthenticationError is mapped to EmbeddingAuthenticationError safely."""
    dummy_req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    dummy_resp = httpx.Response(401, request=dummy_req)
    fake_err = openai.AuthenticationError(
        message="Incorrect API key provided: sk-secret12345",
        response=dummy_resp,
        body={"error": {"message": "Incorrect API key provided: sk-secret12345"}},
    )
    fake_client = FakeOpenAIClient(FakeOpenAIEmbeddings(exception_to_raise=fake_err))
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(EmbeddingAuthenticationError) as exc_info:
        adapter.embed_texts(["Test content"])

    # Must NOT leak secret key in exception message
    assert "sk-secret12345" not in str(exc_info.value)
    assert "OpenAI authentication failed" in str(exc_info.value)


def test_provider_exception_rate_limit_error():
    """Verify openai.RateLimitError is mapped to EmbeddingProviderError safely."""
    dummy_req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    dummy_resp = httpx.Response(429, request=dummy_req)
    fake_err = openai.RateLimitError(
        message="Rate limit reached for requests",
        response=dummy_resp,
        body={"error": {"message": "Rate limit reached"}},
    )
    fake_client = FakeOpenAIClient(FakeOpenAIEmbeddings(exception_to_raise=fake_err))
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        adapter.embed_texts(["Test content"])

    assert "rate limit exceeded" in str(exc_info.value)


def test_provider_exception_timeout_error():
    """Verify openai.APITimeoutError is mapped to EmbeddingProviderError safely."""
    dummy_req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    fake_err = openai.APITimeoutError(request=dummy_req)
    fake_client = FakeOpenAIClient(FakeOpenAIEmbeddings(exception_to_raise=fake_err))
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        adapter.embed_texts(["Test content"])

    assert "timed out" in str(exc_info.value)


def test_provider_exception_connection_error():
    """Verify openai.APIConnectionError is mapped to EmbeddingProviderError safely."""
    dummy_req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    fake_err = openai.APIConnectionError(request=dummy_req)
    fake_client = FakeOpenAIClient(FakeOpenAIEmbeddings(exception_to_raise=fake_err))
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        adapter.embed_texts(["Test content"])

    assert "Failed to connect" in str(exc_info.value)


def test_no_logging_of_sensitive_data(caplog):
    """Verify API keys, prompts, document content, and raw payloads are never logged."""
    sensitive_prompt = "CONFIDENTIAL_PATIENT_RECORD_XYZ"
    fake_client = FakeOpenAIClient(FakeOpenAIEmbeddings(dimension=4))
    adapter = OpenAIEmbeddingAdapter(api_key="sk-test-super-secret-key-12345", client=fake_client)

    with caplog.at_level(logging.DEBUG):
        adapter.embed_texts([sensitive_prompt])

    # Check log records
    for record in caplog.records:
        msg = record.getMessage()
        assert "sk-test-super-secret-key-12345" not in msg
        assert sensitive_prompt not in msg


def test_embed_query_success():
    """Verify embed_query returns a single float vector."""
    fake_client = FakeOpenAIClient(FakeOpenAIEmbeddings(dimension=8))
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    vector = adapter.embed_query("How do I configure timeout?")

    assert isinstance(vector, list)
    assert len(vector) == 8
    assert all(isinstance(x, float) for x in vector)


def test_embed_query_validation():
    """Verify embed_query rejects empty or non-string queries."""
    fake_client = FakeOpenAIClient()
    adapter = OpenAIEmbeddingAdapter(client=fake_client)

    with pytest.raises(ValueError):
        adapter.embed_query("")

    with pytest.raises(ValueError):
        adapter.embed_query("   \t\n  ")

    with pytest.raises(TypeError):
        adapter.embed_query(None)  # type: ignore
