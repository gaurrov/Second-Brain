# Second Brain — Backend Foundation

Foundational backend for a multi-user Personal Knowledge Management (RAG) platform.
This delivery implements: **FastAPI app skeleton, PostgreSQL/SQLAlchemy/Alembic
integration, and the full JWT authentication system** (register, login, refresh,
protected routes). RAG/Qdrant/Groq layers are stubbed in the architecture doc
and will slot into `src/rag/`, `src/vectorstore/`, `src/services/rag_service.py`
per the previously agreed design — not implemented in this pass.

## 1. Project layout (what's implemented so far)

```
src/
├── main.py                        # App factory, middleware, exception handlers, router mount
├── core/
│   ├── config.py                  # Pydantic Settings (single source of truth for env vars)
│   ├── security.py                # bcrypt hashing + JWT create/decode
│   └── exceptions.py              # Framework-agnostic domain exceptions
├── db/
│   ├── session.py                 # Engine + SessionLocal + get_db() dependency
│   └── base_class.py              # Imports all models for Alembic autogenerate
├── models/
│   ├── base.py                    # Declarative Base + UUID/Timestamp mixins
│   └── user_model.py              # User ORM model
├── repositories/
│   ├── base_repository.py         # Generic CRUD
│   └── user_repository.py         # User-specific queries
├── services/
│   └── auth_service.py            # Registration/login/refresh business logic
├── api/
│   ├── deps.py                    # get_db, get_current_user, service factories
│   └── v1/
│       ├── router.py              # Aggregates endpoint routers
│       ├── endpoints/
│       │   ├── auth.py            # POST /api/auth/register, /login, /refresh
│       │   └── users.py           # GET /api/users/profile (protected)
│       └── schemas/
│           ├── auth_schema.py     # Register/Login/Token request-response DTOs
│           └── user_schema.py     # UserResponse DTO
alembic/
├── env.py                         # Wired to settings.sqlalchemy_database_uri + Base.metadata
└── versions/0001_create_users_table.py
tests/
├── conftest.py                    # In-memory SQLite fixtures + TestClient
└── test_auth.py                  # Register/login/refresh/protected-route tests
docker/
├── Dockerfile
└── docker-compose.yml             # Local Postgres for development
```

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# (requirements.txt alone is what ships in the Docker image; requirements-dev.txt
# adds test tooling for local/CI development)

cp .env.example .env
# Edit .env: set a real JWT_SECRET_KEY and Postgres credentials

# Start Postgres locally
docker compose -f docker/docker-compose.yml up -d

# Run migrations
alembic upgrade head

# Run the API
uvicorn src.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## 3. Running tests

```bash
pytest -v
```

Tests run against an in-memory SQLite DB via `tests/conftest.py`, so no live
Postgres instance is required for the auth test suite. `test_auth.py` covers:
registration (success, duplicate email, weak password, invalid username),
login (success, wrong password, unknown email), the protected profile route
(missing token, valid token, garbage token), and refresh-token flow (success,
rejecting an access token presented as a refresh token).

## 4. Design decisions worth knowing

- **`user_id` never comes from the client.** `get_current_user` (in
  `api/deps.py`) is the only place identity is derived — from the verified
  JWT's `sub` claim — and it's threaded through every service/repository call
  from there. This is the isolation mechanism described in the architecture
  doc's Layer 1.
- **Domain exceptions, not HTTPException, in the service layer.** Services
  raise things like `InvalidCredentialsException`; `main.py` maps these to
  HTTP status codes centrally. This keeps `services/` and `repositories/`
  testable without spinning up FastAPI, and keeps HTTP concerns out of
  business logic.
- **User enumeration protection.** Login returns the identical error for
  "no such user" and "wrong password" so an attacker can't use the login
  endpoint to discover which emails are registered.
- **Refresh-token rotation.** `/api/auth/refresh` issues a brand-new
  access+refresh pair rather than just a new access token, which is the
  safer pattern (limits the blast radius of a leaked refresh token).
- **UUID primary keys with server-side defaults.** `gen_random_uuid()`
  (pgcrypto extension, enabled in the first migration) generates IDs at the
  database level, so IDs are safe even on direct DB inserts outside the app.
- **Alembic reads the DB URL from `src.core.config.settings`**, not from a
  hardcoded string in `alembic.ini` — one source of truth for connection info.

## 5. Document Management Module (added in this pass)

Implements upload, listing, retrieval, status polling, and deletion for
user documents (PDF/DOCX/TXT), with the full extract → clean → chunk →
embed → store-in-Qdrant pipeline running as a background task after
upload.

```
src/
├── models/document_model.py         # Document ORM: id, user_id, filename, file_type,
│                                     #   file_path, upload_date (created_at), processing_status, chunk_count
├── api/v1/schemas/document_schema.py # DocumentResponse / DocumentListResponse / DocumentStatusResponse
├── api/v1/endpoints/documents.py     # upload / list / get / status / delete routes
├── repositories/
│   ├── document_repository.py        # Postgres access, always scoped by user_id
│   └── vector_repository.py          # Qdrant access, always scoped by user_id + document_id
├── services/
│   ├── document_service.py           # Upload orchestration, ownership-checked CRUD
│   ├── ingestion_service.py          # extract -> chunk -> embed -> store pipeline
│   └── embedding_service.py          # Sentence-Transformers/BGE wrapper (embed_documents vs embed_query)
├── rag/
│   ├── loaders/                      # pdf_loader.py, docx_loader.py, txt_loader.py + factory
│   ├── cleaners/text_cleaner.py      # Unicode normalization, de-hyphenation, whitespace collapsing
│   └── splitters/text_splitter.py    # LangChain RecursiveCharacterTextSplitter wrapper, page-aware
├── vectorstore/
│   ├── qdrant_client.py              # Singleton client factory
│   └── collection_manager.py         # Collection + payload-index provisioning (idempotent)
└── utils/file_utils.py               # Extension/MIME validation, sanitization, streamed size-limited save
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload a PDF/DOCX/TXT file (202 Accepted; processes async) |
| GET | `/api/documents` | List the current user's documents (paginated) |
| GET | `/api/documents/{id}` | Get one document's metadata |
| GET | `/api/documents/{id}/status` | Poll processing status + chunk_count/error_message |
| DELETE | `/api/documents/{id}` | Delete a document: DB row + Qdrant vectors + file on disk |

### Upload flow

```
POST /api/documents/upload
  -> validate extension + Content-Type (utils/file_utils.py)
  -> stream to disk at storage/uploads/{user_id}/{document_id}/{filename}, enforcing MAX_UPLOAD_SIZE_MB
  -> create `documents` row, status=PENDING
  -> dispatch background task (BackgroundTasks today; swappable to Celery/RQ later)
  -> return 202 + DocumentResponse immediately

[background] process_document_task(document_id):
  -> opens its OWN DB session (see note below)
  -> loader = get_loader(file_type)   # pdf/docx/txt
  -> pages = loader.load(file_path)
  -> chunks = text_splitter.split_pages(pages)   # cleans + splits per page, preserves page_number
  -> embeddings = embedding_service.embed_documents([chunk texts])
  -> vector_repository.upsert_chunks(...)  # payload: {user_id, document_id, filename, chunk_index, page_number, content, timestamp}
  -> document_repository.update_status(COMPLETED, chunk_count=N)  # or FAILED + error_message on any exception
```

**Why the background task opens its own DB session:** FastAPI's `get_db`
dependency closes its session as soon as the request/response cycle
completes — which happens *before* a `BackgroundTasks` callback runs. A
naive implementation that reused the request's session in the background
task would fail with a "session is closed" error. `process_document_task`
therefore opens a fresh `SessionLocal()` and closes it itself — the same
shape it would need as a Celery/RQ task body, so promoting it later is a
drop-in change (see `services/ingestion_service.py`).

### Multi-user isolation in this module

- `DocumentRepository.get_by_id` is deliberately disabled (raises
  `NotImplementedError`) — the only way to fetch a document is
  `get_by_id_for_user(document_id, user_id)`, so it's structurally
  impossible for a future contributor to add an unscoped query.
- `VectorRepository.upsert_chunks` writes `user_id` into every Qdrant
  point's payload; `delete_by_document` filters by **both**
  `document_id` and `user_id` — there is no method that deletes by
  `document_id` alone.
- Files are stored on disk under `{UPLOAD_DIR}/{user_id}/{document_id}/`,
  giving physical directory-level separation in addition to the DB/vector
  checks.
- `document_service.delete_document` always calls `get_document` (which
  raises `DocumentNotFoundException` → HTTP 404) *before* touching
  Qdrant or disk, so a cross-user delete attempt fails closed.

### New dependencies / infrastructure

- **Qdrant** — added to `docker/docker-compose.yml`; collection
  `documents_kb` (configurable) is auto-provisioned on app startup with
  a Cosine-distance vector config and payload indexes on the filterable
  fields: `user_id`, `document_id`, `filename` (keyword) and
  `page_number`, `chunk_index` (integer). `content`/`timestamp` are
  stored but not indexed.
- **Embeddings** — `BAAI/bge-base-en-v1.5` via `sentence-transformers`
  by default (768-dim; update `EMBEDDING_DIMENSION` in `.env` if you
  swap models). The model loads once per process (module-level
  `lru_cache`), embeddings are memoized in a bounded LRU cache
  (`EMBEDDING_CACHE_SIZE`), inputs are deduplicated and encoded in
  sub-batches of `EMBEDDING_BATCH_SIZE`, and inference is serialized
  behind a lock so concurrent ingestion tasks don't race on the shared
  model. Every stored payload carries an ISO-8601 UTC `timestamp`.
- **Upserts** — `VectorRepository.upsert_chunks` writes points in
  batches of `QDRANT_UPSERT_BATCH_SIZE` using Qdrant's columnar `Batch`
  form (parallel ids/vectors/payloads) to keep peak memory flat for
  large documents.
- **Retrieval** — `VectorRepository.search` runs cosine-similarity
  search always filtered by `user_id` (with optional `document_id` /
  `page_number` / `score_threshold` narrowing) and returns typed
  `SearchResult` objects.
- New env vars: `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB`, `QDRANT_*`,
  `EMBEDDING_*`, `CHUNK_SIZE`, `CHUNK_OVERLAP` — see `.env.example`.
- Migration `0002_create_documents_table.py` adds the `documents` table.

### Tests

`tests/test_documents.py` covers upload (success, unsupported type,
empty file, unauthenticated), listing/isolation across two users,
cross-user 404s on get/delete, and 404 on a nonexistent document. The
embedding model and Qdrant client are monkeypatched out so these tests
run fast and don't require a live Qdrant instance or downloading model
weights — they verify the API/service/repository wiring and isolation
guarantees, not raw ML inference quality.

`tests/test_embedding_service.py` unit-tests `EmbeddingService` with an
injected fake model: deduplication, LRU caching/eviction, sub-batch
splitting, the query-instruction prefix, and singleton model loading.

`tests/test_vector_repository.py` is a real integration suite that runs
against an in-process Qdrant (`QdrantClient(":memory:")`) — no Docker
needed. It verifies collection provisioning, batch upserts, the full
payload contract (including `timestamp`), delete/count scoping, and
search ordering/limits/filters/score-thresholds, plus the multi-user
isolation guarantees.

For end-to-end verification against a real Qdrant server and real model
weights (skipped by default):

```bash
docker compose -f docker/docker-compose.yml up -d
pytest --run-live tests/test_live_integration.py
```

## 6. Still on the horizon

- RAG chat orchestration + Groq LLM generation (conversations/messages
  flow, `rag_service.py`, `llm_service.py`, `rag/chains/`). Vector
  retrieval itself is implemented (`VectorRepository.search`); what
  remains is the chain that turns retrieved chunks into an answer.
- Reranking of retrieved chunks
- Promoting `BackgroundTasks` to Celery/RQ for production-grade async
  processing (the ingestion pipeline is already shaped for this)
- Row-Level Security (RLS) in Postgres as defense-in-depth

## 7. Known limitation of this sandbox delivery

This code was written and syntax-checked (`python -m py_compile` on every
file) in an environment without internet/package-registry access, so it could
not be `pip install`-ed and executed end-to-end here. Review the logic before
your first real run; the design follows patterns (FastAPI dependency
injection, SQLAlchemy 2.0 `Mapped`/`mapped_column` style, python-jose,
passlib/bcrypt) that are stable and idiomatic as of the versions pinned in
`requirements.txt`.
