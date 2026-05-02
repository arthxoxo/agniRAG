from __future__ import annotations

import time

from agni_rag.core.chunking import chunk_text
from agni_rag.core.embeddings import Embedder
from agni_rag.core.llm import LLM
from agni_rag.core.models import Chunk, RagAnswer, ScoredChunk
from agni_rag.core.utils import stable_hash
from agni_rag.core.vector_store import VectorStore


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        *,
        max_chunk_words: int,
        chunk_overlap_words: int,
        default_top_k: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._max_chunk_words = max_chunk_words
        self._chunk_overlap_words = chunk_overlap_words
        self._default_top_k = default_top_k

    def ingest_text(
        self,
        tenant_id: str,
        text: str,
        *,
        source_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> int:
        metadata = metadata or {}
        chunks_text = chunk_text(text, self._max_chunk_words, self._chunk_overlap_words)
        chunks = []
        for chunk_text_value in chunks_text:
            chunk_id = stable_hash(f"{tenant_id}:{source_id}:{chunk_text_value}")
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text_value,
                    source_id=source_id,
                    metadata=metadata,
                )
            )
        if not chunks:
            return 0
        embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.upsert(tenant_id, chunks, embeddings)
        return len(chunks)

    def answer_question(
        self,
        tenant_id: str,
        question: str,
        *,
        top_k: int | None = None,
        max_tokens: int = 256,
    ) -> RagAnswer:
        start = time.perf_counter()
        embedding = self._embedder.embed([question])[0]
        top_k_value = top_k or self._default_top_k
        sources = self._store.query(tenant_id, embedding, top_k_value)
        prompt = _build_prompt(question, sources)
        answer = self._llm.generate(prompt, max_tokens)
        _ = time.perf_counter() - start
        return RagAnswer(answer=answer, sources=sources)


def _build_prompt(question: str, sources: list[ScoredChunk]) -> str:
    context_lines = []
    for idx, item in enumerate(sources, start=1):
        context_lines.append(f"[{idx}] {item.chunk.text}")
    context = "\n".join(context_lines)
    return (
        "You are a helpful assistant. Use the context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
