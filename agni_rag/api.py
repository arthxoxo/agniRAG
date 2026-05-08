from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pathlib import Path

from agni_rag.config import Settings
from agni_rag.core.cache import SimpleTTLCache
from agni_rag.core.embeddings import MockEmbedder
from agni_rag.core.llm import MockLLM
from agni_rag.core.rag import RagPipeline
from agni_rag.core.vector_store import MemoryVectorStore
from agni_rag.core.audit import AuditLogger
from agni_rag.security import SecurityManager

_settings = Settings()
_cache = SimpleTTLCache(
    ttl_seconds=_settings.cache_ttl_seconds,
    max_items=_settings.max_cache_items,
)
_security = SecurityManager(_settings)
_audit = AuditLogger(_settings.audit_log_path) if _settings.audit_log_enabled else None
_pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pipeline
    _pipeline = _create_pipeline()
    _pipeline.warmup()
    yield


app = FastAPI(title="agniRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_UI_PATH = Path(__file__).with_name("ui.html")


class IngestRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source_id: str | None = None
    metadata: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    status: str
    queued: bool = True
    chunks_ingested: int | None = None


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    max_tokens: int = 32
    use_cache: bool = True


class SourceOut(BaseModel):
    chunk_id: str
    source_id: str | None
    score: float
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    latency_ms: int
    latency_breakdown_ms: dict[str, int] | None = None


def _create_pipeline() -> RagPipeline:
    if _settings.vector_backend == "qdrant":
        from agni_rag.integrations.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(
            url=_settings.qdrant_url,
            api_key=_settings.qdrant_api_key,
            vector_dim=_settings.embed_dim,
            collection_prefix=_settings.qdrant_collection_prefix,
        )
    else:
        store = MemoryVectorStore()

    if _settings.embedder_backend == "sentence-transformers":
        from agni_rag.integrations.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
        )

        embedder = SentenceTransformerEmbedder(
            model_name=_settings.embedder_model_name,
            cache_size=_settings.embedder_cache_size,
        )
    else:
        embedder = MockEmbedder(dim=_settings.embed_dim)

    if _settings.llm_backend == "mlx":
        from agni_rag.integrations.mlx_llm import MlxLLM

        llm = MlxLLM(
            model_id=_settings.mlx_model_id,
            max_tokens=_settings.mlx_max_tokens,
            temperature=_settings.mlx_temperature,
            top_p=_settings.mlx_top_p,
        )
    elif _settings.llm_backend == "ollama":
        from agni_rag.integrations.ollama_llm import OllamaLLM

        llm = OllamaLLM(
            model=_settings.ollama_model,
            base_url=_settings.ollama_base_url,
            num_ctx=_settings.ollama_num_ctx,
            num_predict=_settings.ollama_num_predict,
            num_batch=_settings.ollama_num_batch,
            num_thread=_settings.ollama_num_thread,
            num_gpu=_settings.ollama_num_gpu,
            temperature=_settings.ollama_temperature,
            top_p=_settings.ollama_top_p,
            top_k=_settings.ollama_top_k,
            timeout_secs=_settings.ollama_timeout_secs,
        )
    elif _settings.llm_backend == "llama_cpp":
        from agni_rag.integrations.llama_cpp_llm import LlamaCppLLM

        if not _settings.llama_model_path:
            raise RuntimeError("LLAMA_MODEL_PATH is required for llama_cpp backend")
        try:
            llm = LlamaCppLLM(
                model_path=_settings.llama_model_path,
                n_ctx=_settings.llama_n_ctx,
                n_threads=_settings.llama_threads,
                n_gpu_layers=_settings.llama_gpu_layers,
            )
        except ModuleNotFoundError as exc:
            print(
                "Warning: llama_cpp is not installed; falling back to MockLLM. "
                "Install with: pip install llama-cpp-python"
            )
            llm = MockLLM()
    else:
        llm = MockLLM()

    return RagPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        max_chunk_words=_settings.max_chunk_words,
        chunk_overlap_words=_settings.chunk_overlap_words,
        default_top_k=_settings.default_top_k,
    )


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline is not initialized yet")
    return _pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "vector_backend": _settings.vector_backend,
        "embedder_backend": _settings.embedder_backend,
        "llm_backend": _settings.llm_backend,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "agniRAG",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    _security.authorize(request.tenant_id, http_request)

    def run_ingest() -> None:
        start = time.perf_counter()
        try:
            pipeline = _get_pipeline()
            chunks = _ingest_payload(pipeline, request)
            _log_audit(
                event="ingest",
                tenant_id=request.tenant_id,
                status="ok",
                latency_ms=_latency_ms(start),
                extra={"chunks_ingested": chunks, "source_id": request.source_id},
            )
        except HTTPException as exc:
            _log_audit(
                event="ingest",
                tenant_id=request.tenant_id,
                status="error",
                latency_ms=_latency_ms(start),
                extra={"error": exc.detail, "status_code": exc.status_code},
            )
        except Exception as exc:
            _log_audit(
                event="ingest",
                tenant_id=request.tenant_id,
                status="error",
                latency_ms=_latency_ms(start),
                extra={"error": str(exc), "status_code": 500},
            )

    background_tasks.add_task(run_ingest)
    return IngestResponse(status="queued")


def _ingest_payload(pipeline: RagPipeline, request: IngestRequest) -> int:
    return pipeline.ingest_text(
        tenant_id=request.tenant_id,
        text=request.text,
        source_id=request.source_id,
        metadata=request.metadata,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    start = time.perf_counter()
    _security.authorize(request.tenant_id, http_request)
    cache_key = f"{request.tenant_id}:{request.question}:{request.top_k}:{request.max_tokens}"
    if request.use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        pipeline = _get_pipeline()
        answer = pipeline.answer_question(
            tenant_id=request.tenant_id,
            question=request.question,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )
        latency_ms = _latency_ms(start)

        response = QueryResponse(
            answer=answer.answer,
            sources=[
                SourceOut(
                    chunk_id=item.chunk.chunk_id,
                    source_id=item.chunk.source_id,
                    score=item.score,
                    text=item.chunk.text,
                    metadata=item.chunk.metadata,
                )
                for item in answer.sources
            ],
            latency_ms=latency_ms,
            latency_breakdown_ms=answer.timing_ms,
        )

        if request.use_cache:
            _cache.set(cache_key, response)

        _log_audit(
            event="query",
            tenant_id=request.tenant_id,
            status="ok",
            latency_ms=latency_ms,
            extra={"top_k": request.top_k, "use_cache": request.use_cache},
        )
        return response
    except HTTPException as exc:
        _log_audit(
            event="query",
            tenant_id=request.tenant_id,
            status="error",
            latency_ms=_latency_ms(start),
            extra={"error": exc.detail, "status_code": exc.status_code},
        )
        raise
    except Exception as exc:
        _log_audit(
            event="query",
            tenant_id=request.tenant_id,
            status="error",
            latency_ms=_latency_ms(start),
            extra={"error": str(exc), "status_code": 500},
        )
        raise


@app.delete("/kb/{tenant_id}")
def delete_kb(tenant_id: str, http_request: Request) -> dict[str, str]:
    _security.authorize(tenant_id, http_request)
    _get_pipeline().store.delete(tenant_id)
    return {"status": "deleted", "tenant_id": tenant_id}


@app.exception_handler(RuntimeError)
def runtime_error_handler(_request, exc: RuntimeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _latency_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _log_audit(
    *,
    event: str,
    tenant_id: str,
    status: str,
    latency_ms: int,
    extra: dict[str, Any],
) -> None:
    if _audit is None:
        return
    payload = {
        "event": event,
        "tenant_id": tenant_id,
        "status": status,
        "latency_ms": latency_ms,
    }
    payload.update(extra)
    try:
        _audit.log(payload)
    except Exception:
        pass
