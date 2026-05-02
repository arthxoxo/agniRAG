import json
import tempfile
import unittest

from agni_rag.core.rate_limit import TokenBucketRateLimiter
from agni_rag.core.tenants import TenantRegistry


class SecurityTests(unittest.TestCase):
    def test_rate_limiter_blocks_burst(self) -> None:
        limiter = TokenBucketRateLimiter(rps=0.0, burst=1)
        self.assertTrue(limiter.allow("acme"))
        self.assertFalse(limiter.allow("acme"))

    def test_tenant_registry_loads(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            json.dump({"tenants": [{"id": "acme", "api_key": "key"}]}, handle)
            path = handle.name

        registry = TenantRegistry(path)
        tenant = registry.get("acme")
        self.assertIsNotNone(tenant)
        self.assertEqual(tenant.tenant_id, "acme")
        self.assertEqual(tenant.api_key, "key")


if __name__ == "__main__":
    unittest.main()
