from __future__ import annotations

from typing import Any

from agni_rag.core.models import Chunk, ScoredChunk
from agni_rag.core.utils import stable_hash
from agni_rag.core.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        vector_dim: int,
        collection_prefix: str = "tenant_",
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest

        self._client = QdrantClient(url=url, api_key=api_key)
        self._rest = rest
        self._vector_dim = vector_dim
        self._collection_prefix = collection_prefix

    def upsert(self, tenant_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        collection = self._collection_name(tenant_id)
        self._ensure_collection(collection)
        points = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_id": chunk.source_id,
                "metadata": chunk.metadata,
            }
            point_id = stable_hash(f"{tenant_id}:{chunk.chunk_id}")
            points.append(self._rest.PointStruct(id=point_id, vector=vector, payload=payload))
        if points:
            self._client.upsert(collection_name=collection, points=points)

    def query(self, tenant_id: str, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        collection = self._collection_name(tenant_id)
        self._ensure_collection(collection)
        results = self._client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        seen: set[str] = set()
        unique_results = []
        for hit in results:
            payload = hit.payload or {}
            chunk_id = str(payload.get("chunk_id") or hit.id)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique_results.append(hit)

        scored: list[ScoredChunk] = []
        for result in unique_results[:top_k]:
            payload: dict[str, Any] = result.payload or {}
            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id") or result.id),
                text=str(payload.get("text", "")),
                source_id=payload.get("source_id"),
                metadata=payload.get("metadata") or {},
            )
            scored.append(ScoredChunk(chunk=chunk, score=float(result.score)))
        return scored

    def _collection_name(self, tenant_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in tenant_id)
        return f"{self._collection_prefix}{safe}"

    def _ensure_collection(self, name: str) -> None:
        try:
            self._client.get_collection(name)
            return
        except Exception:
            pass

        self._client.create_collection(
            collection_name=name,
            vectors_config=self._rest.VectorParams(
                size=self._vector_dim,
                distance=self._rest.Distance.COSINE,
            ),
            hnsw_config=self._rest.HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=10000,
            ),
        )
