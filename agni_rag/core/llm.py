from __future__ import annotations

import re
from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str:
        ...


class MockLLM:
    def __init__(self) -> None:
        self._question_re = re.compile(r"Question:(.*)")

    def generate(self, prompt: str, max_tokens: int) -> str:
        match = self._question_re.search(prompt)
        question = match.group(1).strip() if match else ""
        return f"Mock answer based on provided context. Question: {question}".strip()
