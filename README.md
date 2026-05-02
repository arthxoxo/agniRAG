# agniRAG

Local, multi-tenant RAG backend scaffold for whitelabel AI agents.

## Quick start (mock backends)

This runs end-to-end without external APIs using mock embeddings and a mock LLM.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn agni_rag.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Example ingest/query:

```bash
curl -X POST http://localhost:8000/ingest \
	-H 'Content-Type: application/json' \
	-H 'X-API-Key: change-me' \
	-d '{"tenant_id":"acme","source_id":"doc-1","text":"AgniRAG keeps data local."}'

curl -X POST http://localhost:8000/query \
	-H 'Content-Type: application/json' \
	-H 'X-API-Key: change-me' \
	-d '{"tenant_id":"acme","question":"Where is data stored?"}'
```

## Qdrant (vector DB)

Start Qdrant locally:

```bash
docker compose up -d
```

Set environment:

```bash
export VECTOR_BACKEND=qdrant
export QDRANT_URL=http://localhost:6333
```

## SentenceTransformers embeddings

Enable real local embeddings:

```bash
export EMBEDDER_BACKEND=sentence-transformers
```

## Local LLM (llama.cpp)

Install the Python bindings and configure the model:

```bash
pip install llama-cpp-python
export LLM_BACKEND=llama_cpp
export LLAMA_MODEL_PATH=/path/to/model.gguf
```

## Configuration

Copy and edit [./.env.example](.env.example) or set environment variables directly.

## Tenant auth + rate limiting + audit

Auth is controlled by `AUTH_ENABLED`. Provide tenant API keys in `TENANT_CONFIG_PATH`.
See [./tenants.example.json](tenants.example.json) for the expected format.

Rate limiting uses a token bucket. Defaults:

- `RATE_LIMIT_RPS`
- `RATE_LIMIT_BURST`

Audit logging writes JSON lines to `AUDIT_LOG_PATH` when `AUDIT_LOG_ENABLED=true`.

## Benchmark (mock backends)

```bash
python scripts/benchmark.py
```