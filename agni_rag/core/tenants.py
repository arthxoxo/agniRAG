from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TenantConfig:
    tenant_id: str
    api_key: str
    rate_limit_rps: float | None = None
    rate_limit_burst: int | None = None


class TenantRegistry:
    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._cache_mtime: float | None = None
        self._tenants: dict[str, TenantConfig] = {}
        self._load()

    def get(self, tenant_id: str) -> TenantConfig | None:
        self._reload_if_changed()
        return self._tenants.get(tenant_id)

    def _reload_if_changed(self) -> None:
        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            return
        if self._cache_mtime is None or mtime > self._cache_mtime:
            self._load()

    def _load(self) -> None:
        try:
            with open(self._config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self._tenants = {}
            self._cache_mtime = None
            return
        tenants = {}
        for item in payload.get("tenants", []):
            config = _parse_tenant(item)
            if config:
                tenants[config.tenant_id] = config
        self._tenants = tenants
        try:
            self._cache_mtime = os.path.getmtime(self._config_path)
        except OSError:
            self._cache_mtime = None


def _parse_tenant(data: dict[str, Any]) -> TenantConfig | None:
    tenant_id = str(data.get("id", "")).strip()
    api_key = str(data.get("api_key", "")).strip()
    if not tenant_id or not api_key:
        return None
    rate_limit = data.get("rate_limit", {})
    rps = rate_limit.get("rps")
    burst = rate_limit.get("burst")
    return TenantConfig(
        tenant_id=tenant_id,
        api_key=api_key,
        rate_limit_rps=float(rps) if rps is not None else None,
        rate_limit_burst=int(burst) if burst is not None else None,
    )
