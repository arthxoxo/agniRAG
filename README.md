# agniRAG

Local, multi-tenant RAG backend scaffold for whitelabel AI agents.

## Quick start (mock backends)

This runs end-to-end without external APIs using mock embeddings and a mock LLM.
Use Python `3.10` to `3.12` (Python `3.13+` can cause dependency resolver backtracking with current NLP packages).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn agni_rag.api:app --host 0.0.0.0 --port 8000 --workers 1
```

The single API entrypoint is `agni_rag.api:app`, and the RAG pipeline is loaded at startup via FastAPI lifespan.

Example ingest/query:

Ingest accepts raw text. Ingest is queued and runs in the background.

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

## Local LLM (Ollama)

Install Ollama, pull a model, and point the app at your local server:

```bash
ollama pull phi3:mini
export LLM_BACKEND=ollama
export OLLAMA_MODEL=phi3:mini
export OLLAMA_BASE_URL=http://localhost:11434
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
