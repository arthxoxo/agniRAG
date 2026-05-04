from __future__ import annotations

from collections import OrderedDict

from agni_rag.core.embeddings import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, cache_size: int = 2048) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._cache_size = max(cache_size, 0)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indexes: list[int] = []

        for idx, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is None:
                missing_texts.append(text)
                missing_indexes.append(idx)
            else:
                results[idx] = cached

        if missing_texts:
            vectors = self._model.encode(missing_texts, normalize_embeddings=True)
            for out_idx, vector in enumerate(vectors):
                as_list = vector.tolist()
                input_idx = missing_indexes[out_idx]
                text = texts[input_idx]
                results[input_idx] = as_list
                self._cache_set(text, as_list)

        return [vector for vector in results if vector is not None]

    def _cache_get(self, text: str) -> list[float] | None:
        if self._cache_size <= 0:
            return None
        value = self._cache.get(text)
        if value is None:
            return None
        self._cache.move_to_end(text)
        return value

    def _cache_set(self, text: str, value: list[float]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = value
        self._cache.move_to_end(text)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
