from dataclasses import dataclass
import os


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


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

    ollama_model: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
    ollama_num_predict: int | None = _env_int("OLLAMA_NUM_PREDICT")
    ollama_num_batch: int | None = _env_int("OLLAMA_NUM_BATCH")
    ollama_num_thread: int | None = _env_int("OLLAMA_NUM_THREAD")
    ollama_num_gpu: int | None = _env_int("OLLAMA_NUM_GPU")
    ollama_temperature: float = _env_float("OLLAMA_TEMPERATURE", 0.0) or 0.0
    ollama_top_p: float | None = _env_float("OLLAMA_TOP_P")
    ollama_top_k: int | None = _env_int("OLLAMA_TOP_K")
    ollama_timeout_secs: float = _env_float("OLLAMA_TIMEOUT_SECS", 30.0) or 30.0

    mlx_model_id: str = os.getenv("MLX_MODEL_ID", "mlx-community/gemma-2b-it-4bit")
    mlx_max_tokens: int = int(os.getenv("MLX_MAX_TOKENS", "64"))
    mlx_temperature: float = _env_float("MLX_TEMPERATURE", 0.0) or 0.0
    mlx_top_p: float = _env_float("MLX_TOP_P", 0.9) or 0.9

    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    embedder_model_name: str = os.getenv("EMBEDDER_MODEL_NAME", "all-MiniLM-L6-v2")
    embedder_cache_size: int = int(os.getenv("EMBEDDER_CACHE_SIZE", "2048"))
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
