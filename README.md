# DevGuide — Local Developer Documentation Assistant

DevGuide is a local-first, single-process, single-knowledge-base, unauthenticated Retrieval-Augmented Generation (RAG) API MVP. It allows engineering teams to index and query internal technical documentation (runbooks, architecture decision records, API specifications, and troubleshooting guides) directly on a local developer workstation.

---

## 1. Architectural Scope & Operational Boundaries

DevGuide is designed with minimal operational complexity for local developer use:

- **Single Process & Synchronous Execution:** Built on FastAPI using synchronous `def` route handlers. Blocking operations (SQLite, ChromaDB disk access, OpenAI SDK calls) run in FastAPI's internal threadpool without stalling the asynchronous event loop.
- **Single Unified Knowledge Base:** Manages one local document corpus in cosine space.
- **Dual-Store Architecture:**
  - **Relational Metadata Store:** Python standard library `sqlite3` (`SQLITE_PATH`, single `documents` table with `content_sha256` deduplication).
  - **Vector Store:** Local ChromaDB persistent client (`chromadb.PersistentClient` using `CHROMA_PATH`), collection `devguide_chunks` in cosine space.
- **Provider Models:**
  - **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions, batch size 64).
  - **Generation:** OpenAI Responses API using `gpt-6-luna` (configured via `OPENAI_CHAT_MODEL`).
- **Parsing:** Page-aware extraction for PDF using PyMuPDF (`fitz`), preserving 1-based physical page numbers without splitting or merging chunks across page boundaries. Strict UTF-8 decoding for Markdown and TXT files (`page_number=None`).
- **Ephemeral Upload Processing:** Uploaded files are parsed, chunked, embedded, and indexed in-memory and in temporary buffers. Raw files are discarded immediately after indexing.
- **Local Readiness Probe:** `GET /readyz` checks only local SQLite and ChromaDB availability. It does not require `OPENAI_API_KEY` and does not make external network calls.
- **Explicit Non-Goals (Out of Scope):** OCR, cloud object storage, distributed deployment, multi-tenancy, authentication/RBAC, streaming WebSockets, Celery/Redis background queues, agents, or MCP additions.

---

## 2. Canonical Configuration Reference

Configuration is managed via environment variables and loaded through Pydantic Settings (`app.core.config.Settings`).

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_HOST` | `127.0.0.1` | Local bind address |
| `APP_PORT` | `8000` | Local port |
| `DATA_DIR` | `./data` | Base local storage directory |
| `SQLITE_PATH` | `./data/devguide.db` | SQLite database file location |
| `CHROMA_PATH` | `./data/chroma` | Persistent ChromaDB storage path |
| `OPENAI_API_KEY` | *(empty string)* | OpenAI API key (required only for live paid API calls) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-6-luna` | OpenAI chat generation model |
| `CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks in characters |
| `EMBEDDING_BATCH_SIZE` | `64` | Ingestion batch size for embedding generation |
| `DEFAULT_TOP_K` | `5` | Default number of nearest neighbors retrieved |
| `MAX_TOP_K` | `10` | Maximum allowable `top_k` parameter |
| `MAX_COSINE_DISTANCE` | `0.70` | Initial relevance-filtering heuristic threshold |
| `MAX_UPLOAD_MB` | `10` | Maximum allowable file upload size (10 MB = 10,485,760 bytes) |
| `MAX_CONTEXT_CHARS` | `12000` | Maximum aggregate characters included in generation context |

> **Distance Terminology Rule:**
> Cosine distance is reported as the raw geometric distance metric returned by ChromaDB ($0.0$ for identical vectors, $1.0$ for orthogonal vectors, up to $2.0$ for diametrically opposed vectors). Smaller cosine distances indicate closer vectors in embedding space. It is **never** referred to or transformed into a "confidence percentage" or "probability score".

---

## 3. Environment Setup & Installation

### 3.1 Prerequisites
- Python 3.11 or higher
- `pip` package manager
- Linux, macOS, or WSL2

### 3.2 Installation Steps

1. **Clone the repository and prepare a virtual environment:**
   ```bash
   # Replace this value with your actual Git remote URL.
   REPOSITORY_URL="YOUR_REPOSITORY_URL"
   git clone "$REPOSITORY_URL" DevGuide
   cd DevGuide
   python3 -m venv .venv
   source .venv/bin/activate
   ```


2. **Install dependencies:**
   ```bash
   # Production dependencies
   pip install -r requirements.txt

   # Development and testing dependencies
   pip install -r requirements-dev.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   *Note: For local development and automated testing, `OPENAI_API_KEY` can remain empty. Automated test suites use test doubles and require no API key or external connectivity.*

---

## 4. Running the Application

### 4.1 Start the API Server
Start the service with Uvicorn:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The application initializes SQLite tables and the ChromaDB collection automatically during startup lifespan.

### 4.2 Interactive Documentation
Interactive API exploration is available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema:** `http://127.0.0.1:8000/openapi.json`

---

## 5. Endpoints & Usage Examples

### 5.1 Health & Readiness Checks

#### Liveness Probe (`GET /healthz`)
Verifies the FastAPI process is running. Decoupled from SQLite, ChromaDB, and OpenAI.
```bash
curl -X GET http://127.0.0.1:8000/healthz
```
**Response (200 OK):**
```json
{
  "status": "ok"
}
```

#### Readiness Probe (`GET /readyz`)
Verifies local storage dependencies (SQLite database file and ChromaDB persistent directory). Does not require `OPENAI_API_KEY` and never tests external network connectivity.
```bash
curl -X GET http://127.0.0.1:8000/readyz
```
**Response (200 OK):**
```json
{
  "status": "ready",
  "checks": {
    "sqlite": "ok",
    "chromadb": "ok"
  }
}
```

---

### 5.2 Document Management

#### Upload and Index Document (`POST /api/v1/documents`)
Uploads and indexes a `.pdf`, `.md`, or `.txt` document. Enforces SHA-256 deduplication against SQLite `content_sha256`.
```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -F "file=@sample_docs/quickstart.md"
```
**Response (201 Created):**
```json
{
  "id": "e4b57d62-817a-4cfa-9311-66d483428d01",
  "filename": "quickstart.md",
  "file_type": "md",
  "content_sha256": "b2c9ef8f2c3d102e3b2e3e5a2b0c1f4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d",
  "chunk_count": 3,
  "created_at": "2026-11-24T12:00:00Z"
}
```

**Common Error Responses:**
- `400 Bad Request`: Unsupported file extension, empty file, or unparseable content.
- `409 Conflict`: Document with identical content SHA-256 hash already exists.
- `413 Payload Too Large`: Upload exceeds `MAX_UPLOAD_MB` (10 MB).
- `500 Internal Server Error`: Partial consistency failure (ChromaDB vectors persisted but SQLite registration failed and cleanup failed).
- `503 Service Unavailable`: Upstream OpenAI embedding failure during indexing.

#### List Documents (`GET /api/v1/documents`)
Lists all indexed documents and summary counts.
```bash
curl -X GET http://127.0.0.1:8000/api/v1/documents
```
**Response (200 OK):**
```json
{
  "documents": [
    {
      "id": "e4b57d62-817a-4cfa-9311-66d483428d01",
      "filename": "quickstart.md",
      "file_type": "md",
      "content_sha256": "b2c9ef8f2c3d102e3b2e3e5a2b0c1f4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d",
      "chunk_count": 3,
      "created_at": "2026-11-24T12:00:00Z"
    }
  ],
  "total": 1
}
```

#### Get Document Details (`GET /api/v1/documents/{document_id}`)
Retrieves metadata for a specific document by UUID.
```bash
curl -X GET http://127.0.0.1:8000/api/v1/documents/e4b57d62-817a-4cfa-9311-66d483428d01
```
**Response (200 OK):**
```json
{
  "id": "e4b57d62-817a-4cfa-9311-66d483428d01",
  "filename": "quickstart.md",
  "file_type": "md",
  "content_sha256": "b2c9ef8f2c3d102e3b2e3e5a2b0c1f4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d",
  "chunk_count": 3,
  "created_at": "2026-11-24T12:00:00Z"
}
```

#### Delete Document (`DELETE /api/v1/documents/{document_id}`)
Deletes Chroma vectors first, then deletes SQLite metadata.
```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/e4b57d62-817a-4cfa-9311-66d483428d01
```
**Response (200 OK):**
```json
{
  "id": "e4b57d62-817a-4cfa-9311-66d483428d01",
  "deleted": true,
  "chunks_removed": 3
}
```

*Partial Failure Consistency:* If Chroma vector deletion fails, the operation aborts immediately and preserves SQLite metadata (HTTP 500). If Chroma vector deletion succeeds but SQLite deletion fails, the operation reports a partial consistency failure (HTTP 500).

---

### 5.3 Semantic Retrieval (Generation-Free)

#### Retrieve Relevant Chunks (`POST /api/v1/retrieve`)
Performs vector search in ChromaDB and applies `distance <= 0.70` filtering without invoking the generation model. Does not assign generation-specific source labels (`source_label`).
```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I configure database timeout?",
    "top_k": 5
  }'
```
**Response (200 OK):**
```json
{
  "question": "How do I configure database timeout?",
  "requested_top_k": 5,
  "has_relevant_context": true,
  "retrieved_count": 1,
  "matches": [
    {
      "text": "To resolve database locked errors, increase TASKBOARD_DB_TIMEOUT=30 in your .env configuration file.",
      "document_id": "f8a12c34-90ab-4cde-8123-112233445566",
      "filename": "troubleshooting.md",
      "file_type": "md",
      "chunk_index": 0,
      "page_number": null,
      "distance": 0.28
    }
  ]
}
```

When no chunks pass the threshold:
```json
{
  "question": "What is the automated database backup schedule?",
  "requested_top_k": 5,
  "has_relevant_context": false,
  "retrieved_count": 0,
  "matches": []
}
```

---

### 5.4 End-to-End Question Answering (RAG)

#### Query Documentation (`POST /api/v1/query`)
Orchestrates input validation, vector retrieval, relevance filtering (`distance <= 0.70`), bounded context building (`MAX_CONTEXT_CHARS = 12000`), answer generation via OpenAI Responses API, and structured source attribution.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I start the API server?",
    "top_k": 5
  }'
```

**Response (200 OK — Answer Generated with Citations):**
```json
{
  "question": "How do I start the API server?",
  "answer": "You can start the TaskBoard API server in development mode using `taskboard api start --host 0.0.0.0 --port 8080`, or directly with Uvicorn via `uvicorn taskboard.main:app --host 0.0.0.0 --port 8080 --reload`. [S1]",
  "abstained": false,
  "retrieved_count": 1,
  "sources": [
    {
      "source_label": "S1",
      "document_id": "e4b57d62-817a-4cfa-9311-66d483428d01",
      "filename": "quickstart.md",
      "file_type": "md",
      "page_number": null,
      "chunk_index": 2,
      "distance": 0.22,
      "excerpt": "To start the TaskBoard API server in local development mode with automatic reload: taskboard api start --host 0.0.0.0 --port 8080..."
    }
  ]
}
```

#### Abstention Behavior (Zero-Call Abstention)
When no chunks pass `distance <= 0.70`, the pipeline immediately returns the canonical abstention response **without calling the generation model**:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the automated database backup schedule?",
    "top_k": 5
  }'
```

**Response (200 OK — Abstention):**
```json
{
  "question": "What is the automated database backup schedule?",
  "answer": "I do not have enough relevant information in the documentation to answer your question.",
  "abstained": true,
  "retrieved_count": 0,
  "sources": []
}
```

#### Generated Citation Validation
Before returning a successful grounded response, the pipeline verifies that:
1. Every citation marker in the generated answer matches the exact `[S1]`, `[S2]` format.
2. Every cited label corresponds to an existing application-owned `QuerySource` object in `built_context.sources`. Unknown labels (e.g. `[S99]`) or labels referencing sources omitted from the context budget are rejected.
3. For normal non-abstention answers, at least one valid source citation is required.
4. The exact canonical abstention string (`"I do not have enough relevant information in the documentation to answer your question."`) is permitted without citations.
5. If the model produces invalid, omitted, or missing citations, `QueryService` raises `InvalidGenerationResponseError`, which maps safely to HTTP 500 (`{"detail": "Generation service returned an invalid or empty response"}`).

---

## 6. Evaluation Observations (Sample Corpus Sanity Checks)

A deterministic offline evaluation suite was run against the sample technical documentation corpus (`sample_docs/quickstart.md`, `sample_docs/api-reference.md`, `sample_docs/troubleshooting.md`).

> **Note on Evaluation Scope & Test Embedder Identity:**
> This hand-labelled set serves strictly as an operational sanity check of retrieval grounding, distance filtering, evidence verification, and abstention behavior. It is not an empirical benchmark, and no synthetic precision, recall, or accuracy metrics are claimed.
>
> The reported cosine distances were measured using the offline test-only embedding double (`DeterministicCorpusEmbeddingAdapter` with model identity `test-only-deterministic-corpus-embedder-v1`). They are **not** production OpenAI `text-embedding-3-small` measurements or comparable production retrieval benchmarks.

### Observed Retrieval & Query Behavior

| # | Question | Expected Supporting Document | Answerable? | Abstained? | Model Calls | Retrieved Count | Top-1 Chunk & Observed Cosine Distance |
| :- | :--- | :--- | :-: | :-: | :-: | :-: | :--- |
| 1 | *How do I start the API server?* | `quickstart.md` | Yes | `False` | 1 | 2 | `quickstart.md` (chunk 1, distance: 0.5422) |
| 2 | *How do I initialize the database schema?* | `quickstart.md` | Yes | `False` | 1 | 1 | `quickstart.md` (chunk 1, distance: 0.6566) |
| 3 | *What HTTP status code does the health endpoint return when healthy?* | `quickstart.md` | Yes | `False` | 1 | 1 | `quickstart.md` (chunk 2, distance: 0.6570) |
| 4 | *How do I create a new task using the API?* | `api-reference.md` | Yes | `False` | 1 | 2 | `api-reference.md` (chunk 0, distance: 0.6012) |
| 5 | *How do I resolve sqlite3.OperationalError: database is locked?* | `troubleshooting.md` | Yes | `False` | 1 | 1 | `troubleshooting.md` (chunk 0, distance: 0.6059) |
| 6 | *What is the automated database backup schedule?* | None (absent from corpus) | No | `True` | **0** | 0 | None (no chunks $\le 0.70$) |
| 7 | *What is the data retention policy for completed tasks?* | None (absent from corpus) | No | `True` | **0** | 0 | None (no chunks $\le 0.70$) |

### Key Observations
1. **Source-Aware Evidence Grounding:** Every question classified as answerable retrieved its expected supporting document and passage containing key factual evidence (e.g. `quickstart.md` containing `python -m taskboard.cli db-init` for database initialization).
2. **Zero-Call Abstention:** Questions regarding facts genuinely absent from the corpus (backup schedule, retention policy) produced zero chunks with cosine distance $\le 0.70$. The pipeline abstained immediately and skipped LLM generation calls.
3. **Citation & Metadata Integrity:** Source labels `[S1]`, `[S2]` matched exact retrieved chunk excerpts and were validated before response construction; `page_number` was properly formatted as `null` for Markdown documents and 1-based integers for PDFs.
4. **Context Ceiling:** Context window character enforcement guaranteed that candidate chunks exceeding `MAX_CONTEXT_CHARS = 12000` were cleanly omitted without corrupting prompt formatting.


---

## 7. Safe Error Handling & Status Codes

All errors return a standardized JSON error body matching RFC/FastAPI conventions: `{"detail": "..."}`.

| Code | Meaning | Triggers & Handling in DevGuide |
| :--- | :--- | :--- |
| `200 OK` | Succeeded | Successful query, retrieval, list, or deletion |
| `201 Created` | Created | Document successfully chunked, embedded, and indexed |
| `400 Bad Request` | Client error | Unsupported file extension, empty file, empty question, or `top_k` out of bounds ($[1, 10]$) |
| `404 Not Found` | Not found | Document ID does not exist in SQLite |
| `409 Conflict` | Deduplication | Uploaded document content SHA-256 hash already exists |
| `413 Payload Too Large` | Limit exceeded | Uploaded file size exceeds `MAX_UPLOAD_MB` (10 MB) |
| `422 Unprocessable Entity` | Schema error | Missing required fields, invalid JSON, or non-integer `top_k` |
| `500 Internal Server Error` | Unexpected failure | Storage failure, partial consistency failure, or internal error (sanitized without leaking internal paths or stack traces) |
| `503 Service Unavailable` | Dependency failure | Upstream OpenAI API failure during embedding or generation, connection timeout, or rate limiting |

> **Security & Privacy Invariant:**
> Error responses and logs never expose API keys, internal filesystem paths, model prompts, raw document text, or Python stack traces.

---

## 8. Troubleshooting Guide

### 1. `sqlite3.OperationalError: database is locked`
- **Cause:** Multiple concurrent processes attempting to write to `devguide.db`.
- **Resolution:** DevGuide is designed as a single-process local server. Run Uvicorn with `--workers 1`.

### 2. Upstream OpenAI 503 Errors
- **Symptom:** `POST /api/v1/query` returns HTTP 503 with `"OpenAI rate limit exceeded during answer generation"` or connection timeout.
- **Resolution:** Verify your `OPENAI_API_KEY` in `.env` is valid and has sufficient quota. Note that `GET /readyz` will still return 200 OK because readiness depends only on local storage.

### 3. Missing API Key on Ingestion or Query
- **Symptom:** Request returns HTTP 500 indicating storage or service failure.
- **Resolution:** Check `.env` to ensure `OPENAI_API_KEY` is populated before issuing live ingestion or generation requests.

### 4. Port Conflict (`Address already in use`)
- **Symptom:** Uvicorn fails to start on port 8000.
- **Resolution:** Set `APP_PORT=8001` in `.env` or start Uvicorn with `--port 8001`.

---

## 9. Verification & Automated Testing

All automated tests run completely offline using temporary test directories and deterministic test doubles. No real OpenAI API key or network access is required.

Run the full verification suite:
```bash
# 1. Bytecode compilation check
python -m compileall app tests

# 2. Run comprehensive test suite
pytest -v
```
