from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer

QUERY_PROMPT = "Retrieve images or text relevant to the user's query."


class Embedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_images(self, image_paths: list[str]) -> list[list[float]]: ...


def default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class HFEmbedder:
    """Text- und Bild-Embeddings mit einem multimodalen Hugging-Face-Modell
    (z.B. Qwen3-VL-Embedding), beide landen im selben Vektorraum."""

    def __init__(self, model: str, batch_size: int = 8, device: str | None = None):
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model,
            device=device or default_device(),
            model_kwargs={'torch_dtype': torch.bfloat16},
        )

    def embed_query(self, query: str) -> list[float]:
        # Suchanfragen bekommen eine Retrieval-Instruktion, Dokumente den Standard-Prompt des Modells
        return self._encode([{'text': query}], prompt=QUERY_PROMPT)[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._encode([{'text': t} for t in texts])

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        return self._encode([{'image': p} for p in image_paths])

    def _encode(self, inputs: list[dict[str, str]], prompt: str | None = None) -> list[list[float]]:
        if not inputs:
            return []
        embeddings = self._model.encode(
            inputs,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
