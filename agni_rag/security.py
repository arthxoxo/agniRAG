from __future__ import annotations

from fastapi import HTTPException, Request

from agni_rag.config import Settings
from agni_rag.core.rate_limit import TokenBucketRateLimiter
from agni_rag.core.tenants import TenantRegistry


class SecurityManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = TenantRegistry(settings.tenant_config_path)
        self._limiter = TokenBucketRateLimiter(
            rps=settings.rate_limit_rps,
            burst=settings.rate_limit_burst,
        )
        self._tenant_limiters: dict[str, TokenBucketRateLimiter] = {}

    def authorize(self, tenant_id: str, request: Request) -> None:
        tenant = self._registry.get(tenant_id)
        if self._settings.auth_enabled:
            api_key = request.headers.get("X-API-Key", "")
            if not api_key:
                raise HTTPException(status_code=401, detail="Missing API key")
            if tenant is None or tenant.api_key != api_key:
                raise HTTPException(status_code=403, detail="Invalid API key")

        limiter = self._get_limiter(tenant_id, tenant)
        if not limiter.allow(tenant_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    def _get_limiter(
        self,
        tenant_id: str,
        tenant: object | None,
    ) -> TokenBucketRateLimiter:
        if tenant is None:
            return self._limiter
        rps = getattr(tenant, "rate_limit_rps", None) or self._settings.rate_limit_rps
        burst = getattr(tenant, "rate_limit_burst", None) or self._settings.rate_limit_burst
        if rps == self._settings.rate_limit_rps and burst == self._settings.rate_limit_burst:
            return self._limiter
        limiter = self._tenant_limiters.get(tenant_id)
        if limiter is None:
            limiter = TokenBucketRateLimiter(rps=rps, burst=burst)
            self._tenant_limiters[tenant_id] = limiter
        return limiter
