from __future__ import annotations

import httpx

from agni_rag.core.llm import LLM


class OllamaLLM(LLM):
    def __init__(
        self,
        *,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 1024,
        num_predict: int | None = None,
        num_batch: int | None = None,
        num_thread: int | None = None,
        num_gpu: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout_secs: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._num_batch = num_batch
        self._num_thread = num_thread
        self._num_gpu = num_gpu
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._timeout_secs = timeout_secs

        # Warm up so the first real query is faster.
        self.generate("Hi", max_tokens=1)

    def _options(self, max_tokens: int) -> dict[str, object]:
        options: dict[str, object] = {
            "num_ctx": self._num_ctx,
            "num_predict": self._num_predict or max_tokens,
            "temperature": self._temperature,
            "stop": ["<|end|>", "<|user|>"],
        }
        if self._num_batch is not None:
            options["num_batch"] = self._num_batch
        if self._num_thread is not None:
            options["num_thread"] = self._num_thread
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu
        if self._top_p is not None:
            options["top_p"] = self._top_p
        if self._top_k is not None:
            options["top_k"] = self._top_k
        return options

    def generate(self, prompt: str, max_tokens: int) -> str:
        r = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
                "stream": False,
                "keep_alive": "5m",
                "options": self._options(max_tokens),
            },
            timeout=self._timeout_secs,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
