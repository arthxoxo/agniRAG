from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_id: str | None
    metadata: dict[str, Any]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[ScoredChunk]
