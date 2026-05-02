from __future__ import annotations

import math
from typing import Protocol

from agni_rag.core.models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, tenant_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def query(self, tenant_id: str, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._data: dict[str, list[tuple[Chunk, list[float]]]] = {}

    def upsert(self, tenant_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if tenant_id not in self._data:
            self._data[tenant_id] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            self._data[tenant_id].append((chunk, vector))

    def query(self, tenant_id: str, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        if tenant_id not in self._data:
            return []
        scored = []
        for chunk, vector in self._data[tenant_id]:
            score = _cosine_similarity(embedding, vector)
            scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
