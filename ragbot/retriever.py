from typing import Any

from ragbot.embeddings import Embedder
from ragbot.models import RetrievedChunk
from ragbot.reranker import Reranker
from ragbot.vectorstore import VectorStore


class Retriever:
    """Holt `top_k` Kandidaten per Vektorsuche und lässt davon (falls ein Reranker
    gesetzt ist) die besten `rerank_top_n` mit einem Score von mindestens
    `rerank_min_score` übrig."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 16,
        reranker: Reranker | None = None,
        rerank_top_n: int = 4,
        rerank_min_score: float | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.rerank_min_score = rerank_min_score

    def retrieve(self, query: str, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        # Erweiterungspunkte: Query-Rewriting, Distanz-Schwellwert
        if self.store.count() == 0:
            return []
        candidates = self.store.query(self.embedder.embed_query(query), k=self.top_k, where=where)
        if self.reranker is None:
            return candidates
        return self.reranker.rerank(query, candidates, self.rerank_top_n, self.rerank_min_score)
