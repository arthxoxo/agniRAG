from __future__ import annotations

import time


class TokenBucketRateLimiter:
    def __init__(self, rps: float, burst: int) -> None:
        self._rps = max(rps, 0.0)
        self._burst = max(burst, 1)
        self._state: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._state.get(key, (float(self._burst), now))
        if self._rps > 0:
            tokens = min(float(self._burst), tokens + (now - last) * self._rps)
        if tokens < 1.0:
            self._state[key] = (tokens, now)
            return False
        tokens -= 1.0
        self._state[key] = (tokens, now)
        return True
