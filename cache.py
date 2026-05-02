import time


class SimpleTTLCache:
    def __init__(self, ttl_seconds: int, max_items: int = 1024) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if time.time() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value):
        if len(self._store) >= self._max_items:
            self._store.pop(next(iter(self._store)))
        self._store[key] = (time.time() + self._ttl_seconds, value)
