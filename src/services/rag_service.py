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
import logging
import uuid
from dataclasses import dataclass
from typing import Sequence

from src.core.config import settings
from src.core.constants import MessageRole
from src.core.exceptions import ConversationNotFoundException
from src.models.conversation_model import Conversation
from src.models.message_model import Message
from src.rag.chains.injection_guard import PromptInjectionGuard
from src.rag.chains.prompt_builder import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    HistoryItem,
    PromptBuilder,
)
from src.rag.context.compressor import CompressedContext, ContextCompressor
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
        top_k: int | None = None,
        score_threshold: float | None = None,
        rerank_enabled: bool | None = None,
        rerank_top_k: int | None = None,
        history_limit: int | None = None,
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

        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.score_threshold = (
            score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD
        )
        self.rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else settings.RERANK_ENABLED
        )
        self.rerank_top_k = rerank_top_k or settings.RERANK_TOP_K
        self.history_limit = history_limit or settings.CONVERSATION_HISTORY_LIMIT

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
        history = self._load_history(conversation, user_id)

        # 3. Embed the query.
        query_vector = self.embedding_service.embed_query(cleaned)

        # 4. Semantic retrieval - always scoped to `user_id` by the repository.
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

        # 5. Compress the context into a character budget.
        compressed = self.compressor.compress(results)

        # 6. Generate the answer (or refuse without calling the LLM).
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

        # 7. Persist the exchange (user + assistant messages).
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
    # Helpers
    # ------------------------------------------------------------------
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
        return [
            HistoryItem(role=message.role.value, content=message.content)
            for message in messages
        ]

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
    )
