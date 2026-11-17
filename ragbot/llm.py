from collections.abc import Iterator
from typing import Any, Protocol

import ollama

from ragbot.models import Message


class LLM(Protocol):
    def chat(self, messages: list[Message]) -> str: ...

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class OllamaLLM:
    """Chat über Ollama. Das Kontextfenster wird beim Start des Servers gesetzt
    (OLLAMA_CONTEXT_LENGTH), nicht pro Anfrage."""

    def __init__(self, model: str, temperature: float = 0.2, think: bool = False):
        self.model = model
        self.think = think
        self.options = {'temperature': temperature}

    def chat(self, messages: list[Message]) -> str:
        response = ollama.chat(**self._request(messages))
        return response.message.content or ''

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for part in ollama.chat(**self._request(messages), stream=True):
            if part.message.content:
                yield part.message.content

    def _request(self, messages: list[Message]) -> dict[str, Any]:
        return {
            'model': self.model,
            'messages': [m.to_dict() for m in messages],
            'think': self.think,
            'options': self.options,
        }
