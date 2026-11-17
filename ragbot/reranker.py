from typing import Protocol

import torch
from sentence_transformers import CrossEncoder

from ragbot.embeddings import QUERY_PROMPT, default_device
from ragbot.models import Modality, RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int, min_score: float | None = None,
    ) -> list[RetrievedChunk]: ...


class HFReranker:
    """Bewertet Query-Dokument-Paare mit einem multimodalen Cross-Encoder
    (z.B. Qwen3-VL-Reranker), der Text- und Bild-Chunks gleichermaßen versteht."""

    def __init__(self, model: str, batch_size: int = 4, device: str | None = None):
        self.batch_size = batch_size
        self._model = CrossEncoder(
            model,
            device=device or default_device(),
            model_kwargs={'torch_dtype': torch.bfloat16},
        )

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int, min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, _as_document(rc)) for rc in chunks]
        scores = self._model.predict(pairs, prompt=QUERY_PROMPT, batch_size=self.batch_size)
        for rc, score in zip(chunks, scores):
            rc.score = float(score)
        if min_score is not None:
            chunks = [rc for rc in chunks if rc.score >= min_score]
        return sorted(chunks, key=lambda rc: rc.score, reverse=True)[:top_n]


def _as_document(rc: RetrievedChunk) -> dict[str, str]:
    if rc.chunk.modality is Modality.IMAGE and rc.chunk.image_path:
        return {'image': rc.chunk.image_path}
    return {'text': rc.chunk.content}
