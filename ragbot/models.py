from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal['system', 'user', 'assistant']


class Modality(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'


@dataclass
class Chunk:
    """Eine Einheit, die in der Vektor-DB landet.

    Bei Bildern ist `image_path` gesetzt und `content` eine (optionale) Beschreibung,
    die dem LLM als Text mitgegeben wird.
    """
    id: str
    content: str
    modality: Modality = Modality.TEXT
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    distance: float
    # Relevanz-Score des Rerankers (höher = relevanter), None ohne Reranking
    score: float | None = None


@dataclass
class Message:
    role: Role
    content: str
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        msg: dict[str, Any] = {'role': self.role, 'content': self.content}
        if self.images:
            msg['images'] = self.images
        return msg
