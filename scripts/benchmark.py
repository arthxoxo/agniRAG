import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agni_rag.config import Settings
from agni_rag.core.embeddings import MockEmbedder
from agni_rag.core.llm import MockLLM
from agni_rag.core.rag import RagPipeline
from agni_rag.core.vector_store import MemoryVectorStore


def main() -> None:
    settings = Settings()
    pipeline = RagPipeline(
        embedder=MockEmbedder(dim=settings.embed_dim),
        store=MemoryVectorStore(),
        llm=MockLLM(),
        max_chunk_words=settings.max_chunk_words,
        chunk_overlap_words=settings.chunk_overlap_words,
        default_top_k=settings.default_top_k,
    )

    pipeline.ingest_text(
        tenant_id="acme",
        text="AgniRAG keeps tenant data in a local vector database. "
        "Latency targets are 300 to 500 milliseconds.",
        source_id="bench-1",
    )

    start = time.perf_counter()
    answer = pipeline.answer_question("acme", "What is the latency target?")
    latency_ms = int((time.perf_counter() - start) * 1000)

    print(f"Answer: {answer.answer}")
    print(f"Latency: {latency_ms}ms")


if __name__ == "__main__":
    main()
