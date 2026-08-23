"""
RAG service - orchestrates the full retrieval-augmented generation flow:

    Question
        |
        v
    [guard] validate question (prompt-injection defense)
        |
        v
    [embed] embed_query() -> query vector
        |
        v
    [search] VectorRepository.search(query, user_id, top-K, score_threshold)
        |            ^ always filtered by user_id (cross-user retrieval
        |              is structurally impossible)
        v
    [rerank] optional cross-encoder reranking (RERANK_ENABLED)
        |
        v
    [compress] ContextCompressor -> budgeted, deduped context
        |
        v
    [prompt] PromptBuilder -> (system_prompt, user_prompt)
        |
        v
    [llm] LLMService.complete() -> answer
        |
        v
    [save] Conversation + user/assistant messages persisted (user-scoped)

If retrieval returns nothing relevant (empty after the score threshold),
the service returns a polite refusal WITHOUT calling the LLM - the model
can never hallucinate an answer for context it was never given.

`user_id` comes only from the authenticated request (via deps), never
from the client, and is threaded into every repository/vector call.
"""
import hashlib
import logging
import uuid
from dataclasses import dataclass, asdict
from typing import Iterator, Sequence

from src.core.config import settings
from src.core.constants import MessageRole
from src.core.exceptions import ConversationNotFoundException
from src.models.conversation_model import Conversation
from src.models.message_model import Message
from src.rag.chains.injection_guard import PromptInjectionGuard
from src.rag.chains.intent_classifier import is_conversational
from src.rag.chains.prompt_builder import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    HistoryItem,
    PromptBuilder,
)
from src.rag.chains.query_router import QueryRouter, Route
from src.rag.context.compressor import CompressedContext, ContextCompressor, ContextChunk
from src.rag.rerankers.base import IdentityReranker, Reranker
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.vector_repository import VectorRepository
from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMMessage, LLMService

logger = logging.getLogger("second_brain.rag")


@dataclass(frozen=True)
class SourceRef:
    """Provenance for a retrieved chunk surfaced to the API consumer."""

    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    score: float
    snippet: str


@dataclass(frozen=True)
class RAGResult:
    answer: str
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    refused: bool
    sources: list[SourceRef]


class RAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        llm_service: LLMService,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
        *,
        compressor: ContextCompressor | None = None,
        prompt_builder: PromptBuilder | None = None,
        guard: PromptInjectionGuard | None = None,
        reranker: Reranker | None = None,
        query_router: QueryRouter | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        rerank_enabled: bool | None = None,
        rerank_top_k: int | None = None,
        history_limit: int | None = None,
        history_max_characters: int | None = None,
        answer_cache=None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.llm_service = llm_service
        self.conversation_repository = conversation_repository
        self.message_repository = message_repository

        self.compressor = compressor or ContextCompressor()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.guard = guard or PromptInjectionGuard()
        self.reranker = reranker or IdentityReranker()
        self.query_router = query_router or QueryRouter()

        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.score_threshold = (
            score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD
        )
        self.rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else settings.RERANK_ENABLED
        )
        self.rerank_top_k = rerank_top_k or settings.RERANK_TOP_K
        self.history_limit = history_limit or settings.CONVERSATION_HISTORY_LIMIT
        if history_max_characters is not None:
            self.history_max_characters = history_max_characters
        else:
            self.history_max_characters = settings.CONVERSATION_HISTORY_MAX_CHARACTERS

        self._answer_cache = answer_cache
        if self._answer_cache is None and settings.RAG_CACHE_ENABLED:
            from src.core.redis_client import get_cache

            self._answer_cache = get_cache()

    def answer(
        self,
        question: str,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None = None,
    ) -> RAGResult:
        """Run the full RAG pipeline and persist the exchange."""
        # 1. Validate/sanitize the question (raises on injection patterns).
        cleaned = self.guard.validate_question(question)

        # 2. Resolve the conversation, enforcing ownership.
        conversation = self._resolve_conversation(conversation_id, user_id)

        # 3. Load history for routing decision (shared by both routes).
        history = self._load_history(conversation, user_id)

        # 4. Route: CHAT skips Qdrant entirely; DOCUMENT runs the full pipeline.
        routing = self.query_router.route(cleaned, history)
        if routing.route is Route.CHAT:
            return self._answer_chat(cleaned, user_id, conversation, history)

        # --- DOCUMENT route below (existing pipeline) ---

        # 4b. Optional Redis answer cache (RAG_CACHE_ENABLED): repeated
        # questions skip embedding + retrieval + LLM entirely, but the
        # exchange is still persisted with the cached provenance.
        cache_key = self._answer_cache_key(user_id, cleaned) if self._answer_cache else None
        cached = self._answer_cache.get_json(cache_key) if cache_key else None

        if cached is not None:
            answer = cached["answer"]
            refused = bool(cached["refused"])
            chunks = [ContextChunk(**chunk) for chunk in cached["chunks"]]
            sources = [self._to_source_ref(chunk) for chunk in chunks]
            logger.info(
                "RAG answer served from cache user=%s conversation=%s refused=%s sources=%d",
                user_id, conversation.id if conversation else None, refused, len(sources),
            )
        else:
            # 5. Embed the query.
            query_vector = self.embedding_service.embed_query(cleaned)

            # 6. Semantic retrieval - always scoped to `user_id` by the repository.
            candidate_limit = self.top_k * 2 if self.rerank_enabled else self.top_k
            results = self.vector_repository.search(
                query_vector,
                user_id,
                limit=candidate_limit,
                score_threshold=self.score_threshold,
            )
            results = self.reranker.rerank(
                cleaned,
                results,
                top_k=self.rerank_top_k if self.rerank_enabled else self.top_k,
            )

            # 6b. Scan retrieved context for injection (uploads are untrusted).
            self._scan_context_for_injection(results, user_id)

            # 7. Compress the context into a character budget.
            compressed = self.compressor.compress(results)

            # 8. Generate the answer (or refuse without calling the LLM).
            if not compressed.chunks:
                answer = INSUFFICIENT_CONTEXT_RESPONSE
                refused = True
                sources: list[SourceRef] = []
                logger.info(
                    "RAG refused answer for user=%s conversation=%s: no relevant context",
                    user_id, conversation.id if conversation else None,
                )
            else:
                system_prompt, user_prompt = self.prompt_builder.build(
                    cleaned, compressed, history
                )
                answer = self.llm_service.complete(
                    [
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ]
                )
                refused = False
                sources = [self._to_source_ref(chunk) for chunk in compressed.chunks]

            if cache_key is not None:
                self._answer_cache.set_json(
                    cache_key,
                    {
                        "answer": answer,
                        "refused": refused,
                        "chunks": [asdict(chunk) for chunk in compressed.chunks],
                    },
                    ttl=settings.RAG_CACHE_TTL_SECONDS,
                )

        # 9. Persist the exchange (user + assistant messages).
        conversation = self._ensure_conversation(conversation, cleaned, user_id)
        user_message = self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=cleaned,
            )
        )
        assistant_message = self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=answer,
                retrieval_metadata=self._serialize_sources(sources),
            )
        )

        logger.info(
            "RAG answer generated for user=%s conversation=%s refused=%s sources=%d",
            user_id, conversation.id, refused, len(sources),
        )
        return RAGResult(
            answer=answer,
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            refused=refused,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # CHAT route — conversational reply without document retrieval
    # ------------------------------------------------------------------
    def _answer_chat(
        self,
        cleaned: str,
        user_id: uuid.UUID,
        conversation,
        history: list[HistoryItem],
    ) -> RAGResult:
        """Handle a CHAT-routed message: call LLM directly, skip Qdrant."""
        system_prompt, user_prompt = self.prompt_builder.build_conversational(cleaned, history)
        answer = self.llm_service.complete([
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ])

        sources: list[SourceRef] = []
        logger.info(
            "RAG chat reply for user=%s conversation=%s (no retrieval needed)",
            user_id, conversation.id if conversation else None,
        )

        conversation = self._ensure_conversation(conversation, cleaned, user_id)
        user_message = self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=cleaned,
            )
        )
        assistant_message = self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=answer,
                retrieval_metadata=None,
            )
        )

        return RAGResult(
            answer=answer,
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            refused=False,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Streaming CHAT route
    # ------------------------------------------------------------------
    def _answer_stream_chat(
        self,
        cleaned: str,
        user_id: uuid.UUID,
        conversation,
        history: list[HistoryItem],
    ) -> Iterator[dict]:
        """Handle a CHAT-routed streaming message: LLM stream, skip Qdrant."""
        system_prompt, user_prompt = self.prompt_builder.build_conversational(cleaned, history)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        answer_chunks: list[str] = []
        try:
            for chunk in self.llm_service.complete_stream(messages):
                answer_chunks.append(chunk)
                yield {"type": "token", "content": chunk}
        except Exception as exc:
            logger.exception("Streaming LLM call failed for user=%s", user_id)
            yield {"type": "error", "content": "LLM generation failed."}
            return

        answer = "".join(answer_chunks).strip()
        logger.info(
            "RAG stream chat reply for user=%s conversation=%s (no retrieval needed)",
            user_id, conversation.id if conversation else None,
        )

        conversation = self._ensure_conversation(conversation, cleaned, user_id)
        self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=cleaned,
            )
        )
        self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=answer,
                retrieval_metadata=None,
            )
        )

        yield {"type": "sources", "sources": []}

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------
    def answer_stream(
        self,
        question: str,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None = None,
    ) -> Iterator[dict]:
        """Yield SSE-ready dicts for a streaming RAG answer.

        Event types yielded:

        ``{"type": "token", "content": "..."}``
            Incremental text chunk from the LLM.  Multiple events are
            yielded as the model generates output.

        ``{"type": "sources", "sources": [...]}``
            Final event carrying the source citations.

        ``{"type": "error", "content": "..."}``
            Emitted on LLM or retrieval failure.  The stream is closed
            after this event.

        The same retrieval, reranking, compression, and persistence logic
        as ``answer()`` is used — only the LLM call is streamed.
        """
        # 1. Validate / sanitise.
        cleaned = self.guard.validate_question(question)

        # 2. Resolve conversation, enforcing ownership.
        conversation = self._resolve_conversation(conversation_id, user_id)

        # 3. Load history for routing decision.
        history = self._load_history(conversation, user_id)

        # 4. Route: CHAT skips Qdrant entirely; DOCUMENT runs the full pipeline.
        routing = self.query_router.route(cleaned, history)
        if routing.route is Route.CHAT:
            yield from self._answer_stream_chat(cleaned, user_id, conversation, history)
            return

        # --- DOCUMENT route below ---

        # 4b. Optional cache.
        cache_key = self._answer_cache_key(user_id, cleaned) if self._answer_cache else None
        cached = self._answer_cache.get_json(cache_key) if cache_key else None

        if cached is not None:
            # Cache hit — yield the full answer as a single token event.
            answer = cached["answer"]
            refused = bool(cached["refused"])
            chunks = [ContextChunk(**chunk) for chunk in cached["chunks"]]
            sources = [self._to_source_ref(chunk) for chunk in chunks]
            logger.info(
                "RAG stream served from cache user=%s conversation=%s refused=%s sources=%d",
                user_id, conversation.id if conversation else None, refused, len(sources),
            )
            yield {"type": "token", "content": answer}
        else:

            # 3. Embed.
            query_vector = self.embedding_service.embed_query(cleaned)

            # 4. Retrieve — always scoped to user_id.
            candidate_limit = self.top_k * 2 if self.rerank_enabled else self.top_k
            results = self.vector_repository.search(
                query_vector,
                user_id,
                limit=candidate_limit,
                score_threshold=self.score_threshold,
            )
            results = self.reranker.rerank(
                cleaned,
                results,
                top_k=self.rerank_top_k if self.rerank_enabled else self.top_k,
            )

            # 4b. Scan for injection in retrieved context.
            self._scan_context_for_injection(results, user_id)

            # 5. Compress.
            compressed = self.compressor.compress(results)

            # 6. Generate or refuse.
            if not compressed.chunks:
                answer = INSUFFICIENT_CONTEXT_RESPONSE
                refused = True
                sources: list[SourceRef] = []
                logger.info(
                    "RAG stream refused for user=%s conversation=%s: no relevant context",
                    user_id, conversation.id if conversation else None,
                )
                yield {"type": "token", "content": answer}
            else:
                system_prompt, user_prompt = self.prompt_builder.build(
                    cleaned, compressed, history
                )
                messages = [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ]
                answer_chunks: list[str] = []
                try:
                    for chunk in self.llm_service.complete_stream(messages):
                        answer_chunks.append(chunk)
                        yield {"type": "token", "content": chunk}
                except Exception as exc:
                    logger.exception("Streaming LLM call failed for user=%s", user_id)
                    yield {"type": "error", "content": "LLM generation failed."}
                    return

                answer = "".join(answer_chunks).strip()
                refused = False
                sources = [self._to_source_ref(chunk) for chunk in compressed.chunks]

            if cache_key is not None:
                self._answer_cache.set_json(
                    cache_key,
                    {
                        "answer": answer,
                        "refused": refused,
                        "chunks": [asdict(chunk) for chunk in compressed.chunks],
                    },
                    ttl=settings.RAG_CACHE_TTL_SECONDS,
                )

        # 7. Persist the exchange.
        conversation = self._ensure_conversation(conversation, cleaned, user_id)
        user_message = self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=cleaned,
            )
        )
        assistant_message = self.message_repository.create(
            Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=answer,
                retrieval_metadata=self._serialize_sources(sources),
            )
        )

        logger.info(
            "RAG stream completed for user=%s conversation=%s refused=%s sources=%d",
            user_id, conversation.id, refused, len(sources),
        )

        # 8. Final sources event.
        yield {
            "type": "sources",
            "sources": [
                {
                    "document_id": str(source.document_id),
                    "filename": source.filename,
                    "page": source.page_number,
                    "chunk_index": source.chunk_index,
                }
                for source in sources
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _answer_cache_key(user_id: uuid.UUID, cleaned_question: str) -> str:
        """Cache key for a (user, question) answer. User-scoped by construction."""
        digest = hashlib.sha256(f"{user_id}|{cleaned_question}".encode("utf-8")).hexdigest()
        return f"rag:answer:{digest}"

    def _resolve_conversation(
        self, conversation_id: uuid.UUID | None, user_id: uuid.UUID
    ) -> Conversation | None:
        if conversation_id is None:
            return None
        conversation = self.conversation_repository.get_by_id_for_user(conversation_id, user_id)
        if conversation is None:
            raise ConversationNotFoundException()
        return conversation

    def _ensure_conversation(
        self,
        conversation: Conversation | None,
        question: str,
        user_id: uuid.UUID,
    ) -> Conversation:
        if conversation is not None:
            return conversation
        title = (question[:80] or "New conversation")
        return self.conversation_repository.create(
            Conversation(user_id=user_id, title=title)
        )

    def _load_history(
        self, conversation: Conversation | None, user_id: uuid.UUID
    ) -> list[HistoryItem]:
        if conversation is None:
            return []
        messages = self.message_repository.list_for_conversation(
            conversation.id, user_id, limit=self.history_limit, offset=0
        )
        history = [
            HistoryItem(role=message.role.value, content=message.content)
            for message in messages
        ]
        return self._truncate_history(history)

    def _truncate_history(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """Trim history to fit the character budget, keeping the most recent messages.

        Messages are iterated from newest to oldest; once the running
        total exceeds ``self.history_max_characters`` the remaining
        (older) messages are dropped.  A budget of 0 disables the cap.
        """
        if not history or self.history_max_characters <= 0:
            return history
        total = 0
        keep = 0
        for i in range(len(history) - 1, -1, -1):
            total += len(history[i].content)
            if total > self.history_max_characters:
                break
            keep = i
        return history[keep:]

    def _scan_context_for_injection(self, results: Sequence, user_id: uuid.UUID) -> None:
        """Log any injection patterns found inside retrieved chunks.

        Uploaded documents are untrusted input, so retrieved context is
        scanned for the same directive patterns that would reject a user
        question. The prompt-level defenses (system prompt + `<context>`
        delimiters built by PromptBuilder) remain the primary layer; this
        scan exists so an injected document is flagged in the audit trail
        instead of silently relying on the model's behavior.

        Chunks are NOT dropped here: the patterns are high-precision but
        a legitimate document discussing prompt injection legitimately
        contains them, and dropping would silently corrupt retrieval.
        """
        for result in results:
            match = self.guard.scan(result.content)
            if match is None:
                continue
            logger.warning(
                "Context injection pattern detected in retrieved chunk "
                "user=%s document_id=%s filename=%s pattern=%s severity=%s matched=%r",
                user_id,
                result.document_id,
                result.filename,
                match.pattern_key,
                match.severity.value,
                match.matched_text,
            )

    @staticmethod
    def _to_source_ref(chunk) -> SourceRef:
        return SourceRef(
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
            snippet=chunk.content[:500],
        )

    @staticmethod
    def _serialize_sources(sources: Sequence[SourceRef]) -> list[dict] | None:
        """Persistable JSON form of the retrieval provenance for a message."""
        if not sources:
            return None
        return [
            {
                "document_id": source.document_id,
                "filename": source.filename,
                "page_number": source.page_number,
                "chunk_index": source.chunk_index,
                "score": source.score,
                "snippet": source.snippet,
            }
            for source in sources
        ]


def build_rag_service(db) -> RAGService:
    """Factory for use from FastAPI deps / workers."""
    from src.rag.rerankers import build_reranker
    from src.vectorstore.qdrant_client import get_qdrant_client

    return RAGService(
        embedding_service=EmbeddingService(),
        vector_repository=VectorRepository(get_qdrant_client()),
        llm_service=LLMService(),
        conversation_repository=ConversationRepository(db),
        message_repository=MessageRepository(db),
        reranker=build_reranker(),
        query_router=QueryRouter(),
    )
