from __future__ import annotations

from agni_rag.core.llm import LLM


class LlamaCppLLM(LLM):
    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
    ) -> None:
        from llama_cpp import Llama

        try:
            self._llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
            )
        except Exception:
            # Fall back to CPU if GPU initialization fails.
            self._llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,
            )

        # Warm up to reduce first-request latency.
        self._llama(
            "Hi",
            max_tokens=1,
            temperature=0.0,
            top_p=0.9,
            stop=["</s>"],
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        result = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.9,
            stop=["</s>"],
        )
        return result["choices"][0]["text"].strip()
