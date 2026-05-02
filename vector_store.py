from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
)
from qdrant_client.models import HnswConfigDiff
import uuid
import config

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def init_collection():
    client.get_collections()
    print("Qdrant is reachable.")


def _collection_name(kb_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in kb_id)
    return f"{config.COLLECTION_NAME}_{safe}"


def _ensure_collection(kb_id: str) -> str:
    name = _collection_name(kb_id)
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config.VECTOR_DIM,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200,
            ),
        )
        print(f"Collection created: {name}")
    return name


def upsert_chunks(kb_id: str, chunks: list[dict], embedder):
    collection_name = _ensure_collection(kb_id)
    points = []
    for chunk in chunks:
        vector = embedder.encode(chunk["text"]).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "kb_id": kb_id,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["index"],
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search(kb_id: str, query_vector: list[float]) -> list[str]:
    collection_name = _ensure_collection(kb_id)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=config.TOP_K,
        with_payload=True,
    )
    return [hit.payload["text"] for hit in results]


def delete_kb(kb_id: str):
    collection_name = _collection_name(kb_id)
    client.delete_collection(collection_name=collection_name)
