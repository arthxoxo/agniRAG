import httpx
import time

BASE = "http://localhost:8000"

# ingest a test URL
r = httpx.post(
    f"{BASE}/ingest",
    json={
        "kb_id": "test-company",
        "sources": [
            {
                "type": "url",
                "value": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            }
        ],
    },
)
print(r.json())

time.sleep(5)

# query it
r = httpx.post(
    f"{BASE}/query",
    json={
        "kb_id": "test-company",
        "query": "What is retrieval augmented generation?",
    },
)
print(r.json())
