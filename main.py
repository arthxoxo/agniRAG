from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

import config
import ingest
import query
import vector_store
from models import IngestRequest, QueryRequest

# --- app state ---
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — load everything once
    print("Checking Qdrant connectivity...")
    vector_store.init_collection()

    print("Loading embedder...")
    state["embedder"] = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Loading LLM...")
    state["llm"] = Llama(
        model_path=config.LLM_MODEL_PATH,
        n_gpu_layers=config.N_GPU_LAYERS,
        n_ctx=config.N_CTX,
        n_threads=4,
        n_batch=512,
        use_mmap=True,
        verbose=False,
    )
    # warm-up call — first inference is always slow
    state["llm"]("Hello", max_tokens=1)
    print("LLM warmed. Ready.")
    yield
    # shutdown — nothing to clean up


app = FastAPI(lifespan=lifespan)


# --- endpoints ---
@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest, bg: BackgroundTasks):
    def run():
        n = ingest.ingest_sources(
            req.sources,
            req.kb_id,
            state["embedder"],
            vector_store,
        )
        print(f"Ingested {n} chunks for kb_id={req.kb_id}")

    bg.add_task(run)
    return {"status": "ingestion started", "kb_id": req.kb_id}


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    result = query.run_query(
        req.kb_id,
        req.query,
        state["embedder"],
        vector_store,
        state["llm"],
    )
    print(f"Latency: {result['latency_ms']}")
    return result


@app.delete("/kb/{kb_id}")
async def delete_kb(kb_id: str):
    vector_store.delete_kb(kb_id)
    return {"status": "deleted", "kb_id": kb_id}
