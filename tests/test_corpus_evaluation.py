"""Evaluation and sanity checks on supplied sample developer documentation corpus.

Evaluates:
- End-to-end ingestion and indexing of sample_docs/ (quickstart.md, api-reference.md, troubleshooting.md).
- Answerable corpus queries: checks expected vs actual top-K sources, observed distances, and citations.
- Evidence verification: asserts at least one expected supporting document appears in sources and contains key evidence.
- Unanswerable queries: checks proper abstention without model calls when facts are absent.
- Treats hand-labelled questions strictly as sanity checks (no synthetic metrics).
"""

import math
import pathlib
import re
from typing import Any, Optional
import pytest

from app.core.config import Settings
from app.db.chroma import ChromaAdapter
from app.db.repository import DocumentRepository
from app.db.sqlite import init_db
from app.models.schemas import QuerySource
from app.services.generation import (
    CANONICAL_ABSTENTION_ANSWER,
    GenerationResult,
)
from app.services.ingestion import IngestionService
from app.services.query import QueryResult, QueryService
from app.services.retrieval import RetrievalService

STOPWORDS = {
    "what", "is", "the", "a", "an", "for", "in", "of", "to", "how", "do", "i",
    "and", "with", "on", "at", "by", "from", "or", "does", "it", "be",
}


def _stem(w: str) -> str:
    w = w.lower()
    for sfx in ("izations", "ization", "izing", "ized", "ize", "tions", "tion", "ing", "ed", "es", "s"):
        if w.endswith(sfx) and len(w) > len(sfx) + 2:
            return w[: -len(sfx)]
    return w


class DeterministicCorpusEmbeddingAdapter:
    """Deterministic offline test-only embedding adapter for reproducible corpus sanity checks.

    Uses a 512-dimensional deterministic hash projection with section heading weighting
    to simulate semantic alignment without external API keys or paid network calls.
    Unmistakably identified as a test double, NOT the live OpenAI model.
    """

    def __init__(self, dim: int = 512, model: str = "test-only-deterministic-corpus-embedder-v1") -> None:
        self.dim = dim
        self.model = model

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        lines = text.splitlines()
        for line in lines:
            weight = 2.0 if line.strip().startswith("#") else 1.0
            words = [_stem(w) for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", line.lower()) if w not in STOPWORDS]
            for w in words:
                h = 0
                for ch in w:
                    h = (h * 31 + ord(ch)) % self.dim
                vec[h] += weight
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        res = [0.0] * self.dim
        res[0] = 1.0
        return res

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


class DeterministicCorpusGenerationAdapter:
    """Deterministic offline generation double simulating gpt-6-luna evidence-grounded responses."""

    def __init__(self, model: str = "test-only-deterministic-gpt-6-luna") -> None:
        self.model = model
        self.generate_calls: list[dict[str, Any]] = []

    def generate(
        self,
        question: str,
        context: Any,
        system_instructions: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        self.generate_calls.append(
            {
                "question": question,
                "context": context,
            }
        )
        # Formulate grounded answer citing the first labeled evidence source
        first_source = context.sources[0]
        answer = f"According to [{first_source.source_label}], {first_source.excerpt[:120].strip()}..."
        return GenerationResult(
            answer=answer,
            model=model or self.model,
            raw_response_id="eval_resp_001",
        )


@pytest.fixture
def corpus_query_service(tmp_path: pathlib.Path):
    """Fixture initializing temporary SQLite + ChromaDB with ingested sample documents."""
    db_path = tmp_path / "devguide.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path, tmp_path)

    settings = Settings(
        DATA_DIR=tmp_path,
        SQLITE_PATH=db_path,
        CHROMA_PATH=chroma_path,
        DEFAULT_TOP_K=5,
        MAX_TOP_K=10,
        MAX_COSINE_DISTANCE=0.70,
        MAX_CONTEXT_CHARS=12000,
        OPENAI_EMBEDDING_MODEL="test-only-deterministic-corpus-embedder-v1",
        OPENAI_CHAT_MODEL="test-only-deterministic-gpt-6-luna",
    )

    repo = DocumentRepository(db_path)
    chroma = ChromaAdapter(chroma_path, settings=settings)
    chroma.init_collection()
    embedder = DeterministicCorpusEmbeddingAdapter()
    generator = DeterministicCorpusGenerationAdapter()

    ingest_svc = IngestionService(
        repository=repo,
        chroma_adapter=chroma,
        embedding_adapter=embedder,
        settings=settings,
    )

    # Ingest all three sample docs
    sample_docs_dir = pathlib.Path(__file__).parent.parent / "sample_docs"
    for filename in ["quickstart.md", "api-reference.md", "troubleshooting.md"]:
        doc_file = sample_docs_dir / filename
        assert doc_file.exists(), f"Sample doc {filename} missing"
        ingest_svc.ingest_document(filename, doc_file.read_bytes())

    retrieval_svc = RetrievalService(
        chroma_adapter=chroma,
        embedding_adapter=embedder,
        settings=settings,
    )
    query_svc = QueryService(
        retrieval_service=retrieval_svc,
        generation_adapter=generator,
        settings=settings,
    )

    return query_svc, generator, retrieval_svc, settings


# ==============================================================================
# Corpus Sanity Check Evaluation
# ==============================================================================


@pytest.mark.parametrize(
    "question,expected_filename,expected_evidence,expected_answerable",
    [
        (
            "How do I start the API server?",
            "quickstart.md",
            "taskboard api start",
            True,
        ),
        (
            "How do I initialize the database schema?",
            "quickstart.md",
            "python -m taskboard.cli db-init",
            True,
        ),
        (
            "What HTTP status code does the health endpoint return when healthy?",
            "quickstart.md",
            "200",
            True,
        ),
        (
            "How do I create a new task using the API?",
            "api-reference.md",
            "POST /api/v1/tasks",
            True,
        ),
        (
            "How do I resolve sqlite3.OperationalError: database is locked?",
            "troubleshooting.md",
            "TASKBOARD_DB_TIMEOUT=30",
            True,
        ),
        (
            "What is the automated database backup schedule?",
            None,
            None,
            False,
        ),
        (
            "What is the data retention policy for completed tasks?",
            None,
            None,
            False,
        ),
    ],
)
def test_sample_corpus_questions_evaluation(
    corpus_query_service,
    question: str,
    expected_filename: Optional[str],
    expected_evidence: Optional[str],
    expected_answerable: bool,
) -> None:
    """Evaluate questions on sample corpus for grounding, distance thresholding, and abstention."""
    query_svc, generator, retrieval_svc, settings = corpus_query_service

    prev_calls = len(generator.generate_calls)
    result = query_svc.query(question=question, top_k=5)

    if expected_answerable:
        # 1. Answerable queries must not abstain
        assert result.abstained is False, f"Expected answerable query to succeed: {question}"
        assert result.answer != CANONICAL_ABSTENTION_ANSWER
        assert result.retrieved_count > 0
        assert len(result.sources) == result.retrieved_count

        # 2. All returned sources must satisfy distance <= 0.70
        for s in result.sources:
            assert s.distance <= settings.MAX_COSINE_DISTANCE, (
                f"Source distance {s.distance} exceeded {settings.MAX_COSINE_DISTANCE}"
            )
            assert s.source_label.startswith("S")

        # 3. Verify at least one expected supporting document appears in returned sources
        assert expected_filename is not None, "Answerable query must declare expected_filename"
        source_filenames = [s.filename for s in result.sources]
        assert expected_filename in source_filenames, (
            f"Expected {expected_filename} in sources {source_filenames} for '{question}'"
        )

        # 4. Verify at least one matching source excerpt contains the expected evidence phrase
        assert expected_evidence is not None, "Answerable query must declare expected_evidence"
        matching_excerpts = [s.excerpt for s in result.sources if s.filename == expected_filename]
        assert any(expected_evidence in excerpt for excerpt in matching_excerpts), (
            f"Expected evidence '{expected_evidence}' not found in excerpts for {expected_filename}: {matching_excerpts}"
        )

        # 5. Extract citations actually used in generated answer and verify grounding
        cited_labels = re.findall(r"\[(S\d+)\]", result.answer)
        assert len(cited_labels) > 0, f"No citations found in answer: {result.answer}"
        cited_sources = [s for s in result.sources if s.source_label in cited_labels]
        assert any(
            s.filename == expected_filename and expected_evidence in s.excerpt
            for s in cited_sources
        ), (
            f"Expected evidence '{expected_evidence}' from {expected_filename} not found in actually cited sources: "
            f"{[(s.filename, s.source_label, s.excerpt[:60]) for s in cited_sources]}"
        )

        # 6. Generation model must have been called
        assert len(generator.generate_calls) == prev_calls + 1
    else:
        # 1. Unanswerable queries must abstain with canonical string
        assert result.abstained is True, f"Expected unanswerable query to abstain: {question}"
        assert result.answer == CANONICAL_ABSTENTION_ANSWER
        assert result.retrieved_count == 0
        assert result.sources == []

        # 2. CRITICAL: Generation model must NEVER be called on abstention
        assert len(generator.generate_calls) == prev_calls
