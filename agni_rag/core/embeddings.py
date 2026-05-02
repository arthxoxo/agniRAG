from __future__ import annotations

import math
import re
from typing import Protocol

from agni_rag.core.utils import stable_hash


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class MockEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._token_re = re.compile(r"\w+")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = self._token_re.findall(text.lower())
        if not tokens:
            return vec
        for token in tokens:
            idx = int(stable_hash(token), 16) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
