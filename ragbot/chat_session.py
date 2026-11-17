from collections.abc import Iterator
from dataclasses import dataclass, field

from ragbot import prompts
from ragbot.llm import LLM
from ragbot.models import Message, Modality, RetrievedChunk
from ragbot.retriever import Retriever


@dataclass
class ChatResponse:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class ChatSession:
    """Hält den Gesprächsverlauf und orchestriert Retrieval + LLM.

    Im Verlauf wird nur die reine User-Frage gespeichert, nicht der Kontext –
    so bleibt der Verlauf klein und der Kontext wird pro Turn frisch geholt.
    """

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        system_prompt: str = prompts.SYSTEM_PROMPT,
        max_history_turns: int = 10,
    ):
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.history: list[Message] = []

    def ask(self, question: str) -> ChatResponse:
        sources, messages = self._prepare(question)
        answer = self.llm.chat(messages)
        self._remember(question, answer)
        return ChatResponse(answer=answer, sources=sources)

    def ask_stream(self, question: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Liefert die Quellen sofort und die Antwort als Token-Stream.

        Der Verlauf wird erst aktualisiert, wenn der Stream vollständig konsumiert wurde.
        """
        sources, messages = self._prepare(question)

        def generate() -> Iterator[str]:
            parts = []
            for token in self.llm.stream(messages):
                parts.append(token)
                yield token
            self._remember(question, ''.join(parts))

        return sources, generate()

    def reset(self) -> None:
        self.history.clear()

    def _prepare(self, question: str) -> tuple[list[RetrievedChunk], list[Message]]:
        sources = self.retriever.retrieve(question)
        images = [
            rc.chunk.image_path
            for rc in sources
            if rc.chunk.modality is Modality.IMAGE and rc.chunk.image_path
        ]
        current = Message('user', prompts.build_user_prompt(question, sources), images=images)
        messages = [Message('system', self.system_prompt), *self._recent_history(), current]
        return sources, messages

    def _recent_history(self) -> list[Message]:
        if self.max_history_turns <= 0:
            return []
        return self.history[-2 * self.max_history_turns:]

    def _remember(self, question: str, answer: str) -> None:
        self.history.append(Message('user', question))
        self.history.append(Message('assistant', answer))
