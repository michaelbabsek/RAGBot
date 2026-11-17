from typing import Any

import chromadb

from ragbot.models import Chunk, Modality, RetrievedChunk


class VectorStore:
    def __init__(self, path: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[self._to_metadata(c) for c in chunks],
        )

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )
        return [
            RetrievedChunk(chunk=self._from_record(id_, doc, meta), distance=dist)
            for id_, doc, meta, dist in zip(
                result['ids'][0],
                result['documents'][0],
                result['metadatas'][0],
                result['distances'][0],
            )
        ]

    def delete_where(self, where: dict[str, Any]) -> None:
        self._collection.delete(where=where)

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _to_metadata(chunk: Chunk) -> dict[str, Any]:
        meta = {**chunk.metadata, 'modality': chunk.modality.value}
        if chunk.image_path:
            meta['image_path'] = chunk.image_path
        return meta

    @staticmethod
    def _from_record(id_: str, document: str | None, meta: dict[str, Any] | None) -> Chunk:
        meta = dict(meta or {})
        return Chunk(
            id=id_,
            content=document or '',
            modality=Modality(meta.pop('modality', Modality.TEXT.value)),
            image_path=meta.pop('image_path', None),
            metadata=meta,
        )
