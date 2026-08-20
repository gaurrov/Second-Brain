# Second Brain — Multi-User RAG Backend

Production-grade backend for a multi-user Personal Knowledge Management platform.
Users upload documents, which are extracted, chunked, embedded, and stored in a vector database.
A RAG pipeline retrieves relevant chunks and generates grounded answers using a Groq LLM.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115 + Pydantic 2.9 + Uvicorn 0.30 |
| Database | PostgreSQL 15 via SQLAlchemy 2.0 (async-ready) |
| Migrations | Alembic 1.13 |
| Vector Store | Qdrant 1.11 (cosine similarity, 768-dim) |
| Embeddings | sentence-transformers BAAI/bge-base-en-v1.5 |
| LLM | Groq (openai/gpt-oss-120b) |
| Caching / Rate Limiting | Redis 7 |
| Background Workers | Thread pool (in-process) or Redis Queue |
| Observability | Prometheus metrics, structured JSON logging |
| Containerization | Docker multi-stage build, 5-service Compose stack |

## Quick Start

### Local development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # fill JWT_SECRET_KEY, GROQ_API_KEY

docker compose -f docker/docker-compose.yml up -d   # starts Postgres, Qdrant, Redis
alembic upgrade head
uvicorn src.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

### Docker (full stack)

```bash
cp .env.example .env               # fill GROQ_API_KEY, JWT_SECRET_KEY, POSTGRES_PASSWORD
cd docker && docker compose up --build -d
```

Starts all five services (Postgres, Qdrant, Redis, app, rq-worker). Migrations run automatically via `entrypoint.sh`.

## Configuration

All settings live in `src/core/config.py` (Pydantic Settings). Nothing calls `os.environ` directly.
The `.env.example` file lists every variable with sensible defaults. Key groups:

| Group | Variables |
|---|---|
| Database | `POSTGRES_*`, `DATABASE_URL`, `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` |
| JWT / Security | `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `MAX_FAILED_LOGIN_ATTEMPTS`, `ACCOUNT_LOCK_MINUTES` |
| File Storage | `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB` |
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME`, `QDRANT_UPSERT_BATCH_SIZE` |
| Embeddings | `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIMENSION`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_CACHE_SIZE` |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Groq LLM | `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_MAX_TOKENS`, `GROQ_TEMPERATURE` |
| RAG Retrieval | `RETRIEVAL_TOP_K`, `RETRIEVAL_SCORE_THRESHOLD`, `RERANK_ENABLED`, `RERANK_MODEL_NAME` |
| Context | `CONTEXT_MAX_CHARACTERS`, `CONVERSATION_HISTORY_LIMIT`, `CONVERSATION_HISTORY_MAX_CHARACTERS` |
| Redis | `REDIS_ENABLED`, `REDIS_URL`, `CACHE_ENABLED`, `CACHE_TTL_SECONDS` |
| Rate Limiting | `RATE_LIMIT_ENABLED`, `RATE_LIMIT_DEFAULT_LIMIT` (120/min), `RATE_LIMIT_LOGIN_LIMIT` (10/min) |
| Workers | `TASK_WORKER` (`background` / `pool` / `rq`), `WORKER_CONCURRENCY` |
| Observability | `METRICS_ENABLED`, `LOG_FORMAT` (`json` / `text`), `LOG_LEVEL` |

Production validators enforce: JWT secret >= 32 chars, `GROQ_API_KEY` required, no default secret values.

## Project Structure

```
src/
├── main.py                          # App factory, middleware, lifespan, exception handlers
├── core/
│   ├── config.py                    # Pydantic Settings (single source of truth)
│   ├── security.py                  # bcrypt hashing + JWT create/decode
│   ├── health.py                    # Liveness + readiness probes (DB, Qdrant, Redis)
│   ├── metrics.py                   # Prometheus metrics (HTTP, DB, Qdrant, embedding, LLM, Redis)
│   ├── rate_limiter.py              # Fixed-window rate limiter (Redis-backed, memory fallback)
│   ├── redis_client.py              # Connection pool + JSON cache + circuit breaker
│   ├── logging.py                   # Structured JSON (prod) / text (dev) formatters
│   ├── exceptions.py                # Domain exceptions (no HTTP knowledge in services)
│   ├── context.py                   # Per-request correlation ID via ContextVar
│   └── constants.py                 # FileType, ProcessingStatus, MessageRole enums
├── db/
│   ├── session.py                   # Engine + SessionLocal + get_db() + slow query logging
│   └── base_class.py                # Imports all models for Alembic autogenerate
├── models/
│   ├── base.py                      # Declarative Base + UUID/Timestamp mixins
│   ├── user_model.py                # User (roles, brute-force lockout, relationships)
│   ├── document_model.py            # Document (status, chunk_count, composite index)
│   ├── refresh_token_model.py       # RefreshToken (hashed jti only, never raw tokens)
│   ├── conversation_model.py        # Conversation (user_id FK, cascade delete messages)
│   └── message_model.py             # Message (role, content, retrieval_metadata JSON)
├── repositories/
│   ├── base_repository.py           # Generic CRUD[ModelType]
│   ├── user_repository.py           # Email/username lookup, failed login tracking
│   ├── document_repository.py       # Ownership-scoped; get_by_id() disabled (NotImplementedError)
│   ├── conversation_repository.py   # Ownership-scoped; get_by_id() disabled
│   ├── message_repository.py        # Dual-filtered by conversation_id + user_id
│   ├── refresh_token_repository.py  # Hashed token storage, revocation
│   └── vector_repository.py         # Qdrant: upsert, search (user-scoped), delete, count, retry
├── services/
│   ├── auth_service.py              # Register, authenticate, issue/rotate tokens, lockout, logout
│   ├── document_service.py          # Upload orchestration, ownership-checked CRUD
│   ├── ingestion_service.py         # extract → clean → chunk → embed → store pipeline
│   ├── embedding_service.py         # Sentence-Transformers wrapper, L1+L2 cache, dedup, thread-safe
│   ├── conversation_service.py      # Conversation/message CRUD, ownership-scoped
│   ├── llm_service.py               # Groq wrapper, retry, streaming, error translation
│   └── rag_service.py               # Full RAG: inject guard → embed → search → rerank → compress → prompt → LLM → persist
├── rag/
│   ├── chains/
│   │   ├── prompt_builder.py        # System + user prompt construction, anti-hallucination rules
│   │   └── injection_guard.py       # 11-regex prompt-injection defense (question + context sides)
│   ├── context/
│   │   └── compressor.py            # Budgeted context: score-sort, dedup, truncate
│   ├── rerankers/
│   │   ├── base.py                  # Reranker Protocol + IdentityReranker
│   │   └── cross_encoder_reranker.py # Optional cross-encoder reranking
│   ├── loaders/
│   │   ├── base_loader.py           # LoadedPage dataclass, DocumentLoader Protocol
│   │   ├── loader_factory.py        # Maps FileType → loader
│   │   ├── pdf_loader.py            # pypdf, per-page extraction
│   │   ├── docx_loader.py           # python-docx, paragraphs + tables
│   │   ├── txt_loader.py            # UTF-8 with latin-1 fallback
│   │   ├── markdown_loader.py       # Strip code fences/images/HTML
│   │   ├── html_loader.py           # stdlib HTMLParser extraction
│   │   └── csv_loader.py            # Headers + rows as structured text
│   ├── cleaners/
│   │   └── text_cleaner.py          # 7-step: NFKC, ligatures, control chars, de-hyphen, whitespace
│   └── splitters/
│       └── text_splitter.py         # RecursiveCharacterTextSplitter (vendored, no LangChain DLL)
├── vectorstore/
│   ├── qdrant_client.py             # Process-wide singleton (lazy, lru_cache)
│   └── collection_manager.py        # Idempotent collection + payload-index provisioning
├── workers/
│   ├── pool.py                      # Fixed-size daemon thread pool + bounded queue
│   └── rq_worker.py                 # Redis Queue dispatch + standalone worker entry point
├── utils/
│   ├── retry.py                     # Exponential backoff + jitter (sync + async)
│   ├── file_utils.py                # 3-layer validation, streamed save, size enforcement
│   └── timing.py                    # Wall-clock Timer context manager
└── api/
    ├── deps.py                      # DI factories: get_db, get_current_user, service builders
    └── v1/
        ├── router.py                # Aggregates all endpoint routers
        ├── endpoints/
        │   ├── auth.py              # POST register, login, refresh, logout
        │   ├── users.py             # GET profile
        │   ├── documents.py         # POST upload, GET list/get/status, DELETE
        │   └── chat.py              # POST chat, POST chat/stream, GET/DELETE conversations
        └── schemas/
            ├── auth_schema.py       # Register/Login/Token DTOs, username/password validators
            ├── user_schema.py       # UserResponse
            ├── document_schema.py   # DocumentResponse, DocumentListResponse, DocumentStatusResponse
            └── chat_schema.py       # ChatRequest, ChatResponse, Conversation schemas

alembic/
├── env.py                           # Reads settings.sqlalchemy_database_uri
└── versions/
    ├── 0001_create_users_table.py
    ├── 0002_create_documents_table.py
    ├── 0003_add_documents_composite_index.py
    ├── 0004_auth_persistence_hardening.py       # role, brute-force columns, refresh_tokens table
    ├── 0005_create_conversations_and_messages_tables.py
    ├── 0006_add_message_retrieval_metadata.py
    └── 0007_add_message_history_index.py

docker/
├── Dockerfile                       # Multi-stage: builder (torch CPU) → runtime (non-root)
├── docker-compose.yml               # postgres + qdrant + redis + app + rq-worker
└── entrypoint.sh                    # alembic upgrade head → exec "$@"

tests/                               # 26 test files (see Testing section below)
benchmarks/                          # Performance benchmarks (latency, throughput)
```

## API Endpoints

All routes are under `/api/v1`. Authentication is required for every route except register and login.

### Authentication

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/auth/register` | No | Create account (username, email, password) |
| POST | `/api/v1/auth/login` | No | Authenticate → access + refresh tokens |
| POST | `/api/v1/auth/refresh` | No | Exchange refresh token for new pair |
| POST | `/api/v1/auth/logout` | Yes | Revoke all refresh tokens |

Rate limit: **10 requests/minute** for all `/auth/*` endpoints (separate from general limit).

### Users

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/users/profile` | Yes | Get current user's profile |

### Documents

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/documents/upload` | Yes | Upload PDF/DOCX/TXT (202 Accepted; processes async) |
| GET | `/api/v1/documents` | Yes | List current user's documents (paginated) |
| GET | `/api/v1/documents/{id}` | Yes | Get one document's metadata |
| GET | `/api/v1/documents/{id}/status` | Yes | Poll processing status + chunk_count |
| DELETE | `/api/v1/documents/{id}` | Yes | Delete document: DB row + vectors + file on disk |

### Chat & Conversations

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/chat` | Yes | Ask a question → grounded answer + source citations |
| POST | `/api/v1/chat/stream` | Yes | Same as above, streamed as SSE tokens |
| GET | `/api/v1/conversations` | Yes | List current user's conversations (paginated) |
| GET | `/api/v1/conversations/{id}` | Yes | Get conversation with its messages |
| GET | `/api/v1/conversations/{id}/messages` | Yes | List messages in a conversation |
| DELETE | `/api/v1/conversations/{id}` | Yes | Delete a conversation and its messages |

### Infrastructure

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Liveness check |
| GET | `/health/live` | No | Liveness check |
| GET | `/health/ready` | No | Readiness: DB, Qdrant, Redis connectivity + latency |
| GET | `/metrics` | No | Prometheus metrics (HTTP, DB, Qdrant, embedding, Redis, LLM) |

## Testing

### Unit & integration tests (in-memory, no Docker required)

```bash
pytest -v
```

24 test files covering: auth flow, document upload/isolation, conversation CRUD, vector repository (in-process Qdrant), embedding service (fake model), ingestion pipeline, RAG service (fake LLM + lexical embeddings), prompt builder, injection guard, all 6 loaders, text cleaner, text splitter, rate limiter, Redis circuit breaker, retry logic, health probes, worker pool.

### Live integration tests (requires Docker stack running)

```bash
docker compose -f docker/docker-compose.yml up -d
pytest --run-live tests/test_live_integration.py
```

Real Qdrant + real embedding model. Full embed → store → search loop.

### E2E smoke test (standalone script)

```bash
docker compose -f docker/docker-compose.yml up -d
python tests/e2e_test.py
```

32-test suite covering: health checks, multi-user registration/login, document upload/processing, RAG queries (with source grounding), conversation persistence, cross-user IDOR security, SSE streaming, error handling. Each run generates unique credentials. Blocks LLM-dependent tests automatically when `GROQ_API_KEY` is unavailable.

Windows note: set `$env:PYTHONIOENCODING = "utf-8"` before running to handle non-ASCII LLM output.

## Architecture & Design Decisions

### Multi-tenant isolation

Every user-owned resource (documents, conversations, messages, vectors) requires `user_id` at every layer:

- **Repositories**: `get_by_id()` is structurally disabled (`NotImplementedError`) on `DocumentRepository` and `ConversationRepository`. The only fetch path is `get_by_id_for_user()`.
- **Vectors**: Every Qdrant point carries `user_id` in its payload. Search and delete are always filtered by `user_id`.
- **Files on disk**: Stored under `uploads/{user_id}/{document_id}/`, giving physical directory-level separation.
- **Messages**: Carry their own `user_id` in addition to `conversation_id` for double isolation.

### Identity derivation

`get_current_user` (in `api/deps.py`) is the **sole place** user identity is derived — from the verified JWT's `sub` claim. It is never accepted from request bodies or path parameters. This is threaded through every service/repository call.

### Domain exceptions, not HTTPException

Services raise framework-agnostic exceptions (`InvalidCredentialsException`, `DocumentNotFoundException`, etc.). `main.py` maps these to HTTP status codes centrally. Service and repository layers have zero knowledge of HTTP.

### User enumeration protection

Login returns the identical error for "no such user" and "wrong password". A timing-safe dummy check (`verify_password_for_unknown_user`) runs even when the email doesn't exist, so response times are indistinguishable.

### Refresh token rotation

`/api/v1/auth/refresh` issues a brand-new access + refresh pair rather than just a new access token. The old refresh token is revoked. Only the hashed `jti` is stored; raw tokens are never persisted.

### Brute-force protection

After `MAX_FAILED_LOGIN_ATTEMPTS` (default 5) consecutive failures, the account is locked for `ACCOUNT_LOCK_MINUTES` (default 15). Login attempts against a locked account are rejected immediately without checking the password.

### Username validation

Usernames must match `[a-z][a-z0-9_]{2,49}`. Consecutive underscores, trailing underscores, and leading digits are rejected. Passwords require >= 12 characters and must not contain the username or email prefix.

### Three worker backends

`TASK_WORKER` setting selects the document processing dispatch:

| Value | Behavior |
|---|---|
| `background` | FastAPI BackgroundTasks (in-process, dies with request) |
| `pool` | Dedicated daemon thread pool (survives requests, bounded queue) |
| `rq` | Redis Queue (persistent, retry-capable, separate worker process) |

Docker Compose uses `pool` for the app container and runs an `rq-worker` sidecar as well.

### Prompt injection defense

11 heuristic regex patterns on the user question side (high-severity patterns rejected with 400). Context-side patterns are logged but not dropped. `<context>` delimiter tags and strict system prompt rules provide additional defense. Max question length capped at 2000 characters.

### RAG pipeline

```
User question
  → Injection guard (clean + validate)
  → Embed query (BGE model, with query instruction prefix)
  → Vector search (cosine, user-scoped, top-k)
  → Optional reranking (cross-encoder)
  → Context-side injection scan (audit log)
  → Context compression (budget: 6000 chars, dedup threshold 0.95)
  → Prompt construction (system + context + history + question)
  → Groq LLM generation
  → Persist conversation + messages to DB
  → Return answer + source citations
```

Conversations persist across turns. History is truncated by message count (`CONVERSATION_HISTORY_LIMIT`) then by character budget (`CONVERSATION_HISTORY_MAX_CHARACTERS`). The pipeline returns a "insufficient context" refusal without calling the LLM when no relevant chunks are found.

### Observability

- **Prometheus metrics** at `/metrics`: HTTP request count/duration, DB query duration, Qdrant operations, embedding cache hits/misses (L1 + L2), Redis operations, LLM request count, ingestion pipeline counters, worker pool depth.
- **Structured logging**: JSON-lines in production (cloud-shippable with `request_id` correlation), human-readable text in development.
- **Health probes**: `/health/live` (process alive), `/health/ready` (DB + Qdrant + Redis connectivity with latency reporting).
- **Security headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`.

## Migration History

| # | Description |
|---|---|
| 0001 | Users table (UUID PK, username, email, hashed_password, is_active, timestamps). Enables pgcrypto. |
| 0002 | Documents table (user_id FK CASCADE, file metadata, processing status, chunk count) |
| 0003 | Composite index on `(user_id, created_at)` for document list queries |
| 0004 | Auth hardening: `role`, `failed_login_attempts`, `locked_until`, `last_login_at` columns. `refresh_tokens` table. |
| 0005 | Conversations + Messages tables (user_id FK, conversation_id FK, role, content) |
| 0006 | `retrieval_metadata` JSON column on messages |
| 0007 | Composite index on `(conversation_id, user_id, created_at)` for message history queries |

## What's Still on the Horizon

- Reranking of retrieved chunks (cross-encoder, disabled by default via `RERANK_ENABLED`)
- Row-Level Security (RLS) in Postgres as defense-in-depth
- API key authentication for service-to-service access
- Document versioning and incremental re-indexing
- Admin dashboard and usage analytics
