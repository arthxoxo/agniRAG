from __future__ import annotations

from typing import Any

from agni_rag.core.llm import LLM


class MlxLLM(LLM):
    def __init__(
        self,
        *,
        model_id: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        from mlx_lm import load, generate

        self._generate = generate
        self._model, self._tokenizer = load(model_id)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p

        # Warm up the model so the first request is less expensive.
        self.generate("Hi", max_tokens=1)

    def generate(self, prompt: str, max_tokens: int) -> str:
        result = self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=self._temperature,
            top_p=self._top_p,
        )
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()