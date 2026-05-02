from collections import OrderedDict
import time
from typing import Any, OrderedDict as OrderedDictType


class SimpleTTLCache:
    def __init__(self, ttl_seconds: int, max_items: int = 2048) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        self._store: OrderedDictType[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        expires_at, value = self._store[key]
        if time.time() >= expires_at:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        expires_at = time.time() + self._ttl_seconds
        self._store[key] = (expires_at, value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_items:
            self._store.popitem(last=False)
