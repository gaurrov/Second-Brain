"""
End-to-end ingestion pipeline benchmark (live: real model weights).

Runs a representative document through the exact stages the ingestion
service uses — extract -> clean+chunk -> embed -> store — timing each stage
separately. Uses the same loader, splitter, embedding service and vector
repository as ``src/services/ingestion_service.py`` (that service times steps
for its logs/metrics; here the per-stage timings are reported in the
pytest-benchmark table).

PDF is not benchmarked: pypdf is a reader only and the project has no PDF
*writer* dependency. TXT and DOCX cover both loader families (raw text and a
container format) through identical downstream stages.

Run: pytest benchmarks/test_ingestion_pipeline.py --benchmark-only --run-live
"""
import time
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.core.constants import FileType
from src.rag.loaders.loader_factory import get_loader
from src.rag.splitters.text_splitter import TextSplitterService
from src.repositories.vector_repository import VectorRepository
from src.services.embedding_service import EmbeddingService
from src.vectorstore.collection_manager import ensure_collection

pytestmark = pytest.mark.live

PARAGRAPHS = 60


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    import socket

    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        service = EmbeddingService()
        service.embed_documents(["warmup sentence to load model weights"])
        return service
    except Exception as exc:  # noqa: BLE001 - opt-in live suite: skip if weights unavailable
        pytest.skip(f"Embedding model unavailable: {exc}")
    finally:
        socket.setdefaulttimeout(previous)


@pytest.fixture(scope="module")
def repo() -> VectorRepository:
    client = QdrantClient(":memory:")
    ensure_collection(client)
    return VectorRepository(client)


def _write_txt(directory: Path) -> Path:
    path = directory / "sample.txt"
    paragraphs = []
    for index in range(PARAGRAPHS):
        paragraphs.append(
            f"Section {index}. The second brain methodology links notes into a "
            f"graph of ideas, enabling semantic retrieval that surfaces relevant "
            f"context regardless of the original phrasing. Related concepts are "
            f"captured, connected, and reviewed so the knowledge base compounds."
        )
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


def _write_docx(directory: Path) -> Path:
    import docx

    document = docx.Document()
    for index in range(PARAGRAPHS):
        document.add_paragraph(
            f"Section {index}. The second brain methodology links notes into a "
            f"graph of ideas, enabling semantic retrieval that surfaces relevant "
            f"context regardless of the original phrasing. Related concepts are "
            f"captured, connected, and reviewed so the knowledge base compounds."
        )
    path = directory / "sample.docx"
    document.save(path)
    return path


def _run_pipeline(
    file_path: Path,
    file_type: FileType,
    repo: VectorRepository,
    embedding_service: EmbeddingService,
) -> dict:
    timings: dict[str, float] = {}

    start = time.perf_counter()
    pages = get_loader(file_type).load(file_path)
    timings["extract"] = time.perf_counter() - start

    start = time.perf_counter()
    chunks = TextSplitterService().split_pages(pages)
    timings["clean_chunk"] = time.perf_counter() - start

    start = time.perf_counter()
    embeddings = embedding_service.embed_documents([c.content for c in chunks])
    timings["embed"] = time.perf_counter() - start

    start = time.perf_counter()
    written = repo.upsert_chunks(
        user_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=file_path.name,
        chunks=chunks,
        embeddings=embeddings,
    )
    timings["store"] = time.perf_counter() - start

    timings["total"] = sum(timings.values())
    timings["_written"] = float(written)
    timings["_chunks"] = float(len(chunks))
    return timings


def _run_benchmark(benchmark, file_path: Path, file_type: FileType, repo, embedding_service) -> None:
    timings = benchmark.pedantic(
        _run_pipeline,
        args=(file_path, file_type, repo, embedding_service),
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["file_type"] = file_type.value
    benchmark.extra_info["chunks"] = int(timings["_chunks"])
    benchmark.extra_info["written"] = int(timings["_written"])
    for stage in ("extract", "clean_chunk", "embed", "store", "total"):
        benchmark.extra_info[f"{stage}_sec"] = round(timings[stage], 4)
    assert int(timings["_written"]) == int(timings["_chunks"])


def test_ingestion_pipeline_txt(benchmark, embedding_service, repo, tmp_path):
    _run_benchmark(benchmark, _write_txt(tmp_path), FileType.TXT, repo, embedding_service)


def test_ingestion_pipeline_docx(benchmark, embedding_service, repo, tmp_path):
    _run_benchmark(benchmark, _write_docx(tmp_path), FileType.DOCX, repo, embedding_service)
