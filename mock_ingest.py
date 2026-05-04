import httpx, time

BASE = "http://localhost:8000"
TIMEOUT_SECONDS = 120.0
INGEST_WAIT_SECONDS = 12

# 1. ingest a real URL
print("Ingesting...")
r = httpx.post(f"{BASE}/ingest", timeout=TIMEOUT_SECONDS, json={
    "tenant_id": "test-company",
    "sources": [
        {"type": "url", "value": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}
    ]
})
print("Ingest response:", r.json())

# 2. wait for background ingestion to finish
print(f"Waiting {INGEST_WAIT_SECONDS} seconds for ingestion...")
time.sleep(INGEST_WAIT_SECONDS)

# 3. query it
print("Querying...")
r = httpx.post(f"{BASE}/query", timeout=TIMEOUT_SECONDS, json={
    "tenant_id": "test-company",
    "question": "What is retrieval augmented generation?",
    "max_tokens": 64,
    "top_k": 3,
    "use_cache": False
})
print("Query response:", r.json())
