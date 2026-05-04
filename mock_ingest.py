import httpx, time

BASE = "http://localhost:8000"

# 1. ingest a real URL
print("Ingesting...")
r = httpx.post(f"{BASE}/ingest", json={
    "tenant_id": "test-company",
    "sources": [
        {"type": "url", "value": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}
    ]
})
print("Ingest response:", r.json())

# 2. wait for background ingestion to finish
print("Waiting 8 seconds for ingestion...")
time.sleep(8)

# 3. query it
print("Querying...")
r = httpx.post(f"{BASE}/query", json={
    "tenant_id": "test-company",
    "question": "What is retrieval augmented generation?",
    "max_tokens": 150,
    "top_k": 3,
    "use_cache": False
})
print("Query response:", r.json())
