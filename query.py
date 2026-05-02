import time
import config

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using only the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(chunks: list[str], question: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def run_query(kb_id: str, question: str, embedder, vector_store, llm) -> dict:
    t0 = time.monotonic()

    query_vector = embedder.encode(question).tolist()
    t1 = time.monotonic()

    chunks = vector_store.search(kb_id, query_vector)
    t2 = time.monotonic()

    if not chunks:
        return {"answer": "No relevant information found.", "latency_ms": {}}

    prompt = build_prompt(chunks, question)
    t3 = time.monotonic()

    result = llm(
        prompt,
        max_tokens=config.MAX_TOKENS,
        temperature=0.0,
        stop=["Question:", "\n\n\n"],
        echo=False,
    )
    answer = result["choices"][0]["text"].strip()
    t4 = time.monotonic()

    return {
        "answer": answer,
        "latency_ms": {
            "embed": round((t1 - t0) * 1000),
            "search": round((t2 - t1) * 1000),
            "assemble": round((t3 - t2) * 1000),
            "llm": round((t4 - t3) * 1000),
            "total": round((t4 - t0) * 1000),
        },
    }
