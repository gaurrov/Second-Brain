"""
Reranker layer — optional second-stage reordering between vector retrieval
and the LLM prompt.

Pipeline position
-----------------
    Qdrant top-K candidates  (fast, approximate — cosine similarity)
        |
    Reranker                 (slow, accurate — cross-encoder scoring)
        |
    Top-N context            (fed to the LLM)

Latency / quality tradeoff
--------------------------
Vector search (Qdrant) returns results ranked by embedding cosine
similarity, which is fast (~ms) but approximate: two chunks can have
nearly identical embeddings yet differ in how well they actually answer
the question.  A cross-encoder reranker scores each (query, chunk) pair
directly, which is substantially more accurate but requires a forward
pass through a second model (~50-200 ms per candidate on CPU, less with
a GPU).

The practical effect:

  * Disabled (default, ``RERANK_ENABLED=false``):
    Qdrant top-K → compressor → LLM.  Latency is dominated by Qdrant
    retrieval (~1-5 ms) plus embedding (~10-50 ms).  Quality is
    "good enough" for most use cases.

  * Enabled (``RERANK_ENABLED=true``):
    Qdrant top-2K → reranker → top-K → compressor → LLM.  The extra
    reranking step adds ~100-400 ms (proportional to 2K candidates on
    CPU with ``ms-marco-MiniLM-L-6-v2``), but the LLM receives
    strictly better context, which typically improves answer precision
    and reduces hallucination on ambiguous or multi-document queries.

When to enable
--------------
Enable reranking when answer quality matters more than latency — e.g.,
legal/medical Q&A, compliance lookups, or when the document corpus has
many near-duplicate or topically overlapping chunks that confuse
embedding-only retrieval.

Configuration
-------------
``RERANK_ENABLED``       (bool, default false)  — master switch
``RERANK_MODEL_NAME``    (str)  — sentence-transformers cross-encoder
``RETRIEVAL_TOP_K``      (int, default 8) — Qdrant candidate count
``RERANK_TOP_K``          (int, default 4) — final count after reranking

When disabled, ``RETRIEVAL_TOP_K`` results pass through unchanged
(``IdentityReranker``).
"""
from src.core.config import settings
from src.rag.rerankers.base import IdentityReranker, Reranker
from src.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


def build_reranker() -> Reranker:
    """Return the reranker configured by ``RERANK_ENABLED``."""
    if settings.RERANK_ENABLED:
        return CrossEncoderReranker()
    return IdentityReranker()


__all__ = [
    "CrossEncoderReranker",
    "IdentityReranker",
    "Reranker",
    "build_reranker",
]
