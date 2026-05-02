from dataclasses import dataclass
import os


@dataclass
class Settings:
    vector_backend: str = os.getenv("VECTOR_BACKEND", "memory")
    embedder_backend: str = os.getenv("EMBEDDER_BACKEND", "mock")
    llm_backend: str = os.getenv("LLM_BACKEND", "mock")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection_prefix: str = os.getenv("QDRANT_COLLECTION_PREFIX", "tenant_")

    llama_model_path: str = os.getenv("LLAMA_MODEL_PATH", "")
    llama_n_ctx: int = int(os.getenv("LLAMA_N_CTX", "2048"))
    llama_threads: int = int(os.getenv("LLAMA_THREADS", "8"))
    llama_gpu_layers: int = int(os.getenv("LLAMA_GPU_LAYERS", "0"))

    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    max_chunk_words: int = int(os.getenv("MAX_CHUNK_WORDS", "200"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
    default_top_k: int = int(os.getenv("TOP_K", "6"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "120"))
    max_cache_items: int = int(os.getenv("MAX_CACHE_ITEMS", "2048"))

    auth_enabled: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    tenant_config_path: str = os.getenv("TENANT_CONFIG_PATH", "tenants.json")
    rate_limit_rps: float = float(os.getenv("RATE_LIMIT_RPS", "5"))
    rate_limit_burst: int = int(os.getenv("RATE_LIMIT_BURST", "10"))
    audit_log_enabled: bool = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "audit.log")
