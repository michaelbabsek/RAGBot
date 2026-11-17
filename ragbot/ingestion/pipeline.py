from pathlib import Path

from ragbot.embeddings import Embedder
from ragbot.ingestion.chunking import TextChunker
from ragbot.ingestion.pdf_loader import load_pdf
from ragbot.models import Chunk, Modality
from ragbot.vectorstore import VectorStore


class IngestionPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunker: TextChunker,
        image_dir: str,
        render_dpi: int = 150,
    ):
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.image_dir = image_dir
        self.render_dpi = render_dpi

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        pdf_path = Path(pdf_path)
        source = pdf_path.name
        pages = load_pdf(pdf_path, self.image_dir, dpi=self.render_dpi)

        text_chunks: list[Chunk] = []
        image_chunks: list[Chunk] = []
        for page in pages:
            meta = {'source': source, 'page': page.number}
            for i, text in enumerate(self.chunker.split(page.text)):
                text_chunks.append(Chunk(f'{source}:p{page.number}:t{i}', text, metadata=meta))
            image_chunks.append(Chunk(
                f'{source}:p{page.number}:img',
                '',
                modality=Modality.IMAGE,
                image_path=page.image_path,
                metadata=meta,
            ))

        embeddings = self.embedder.embed_texts([c.content for c in text_chunks]) if text_chunks else []
        embeddings += self.embedder.embed_images([c.image_path for c in image_chunks])

        # Alte Version des Dokuments entfernen, damit keine verwaisten Chunks bleiben
        self.store.delete_where({'source': source})
        chunks = text_chunks + image_chunks
        if chunks:
            self.store.upsert(chunks, embeddings)
        return len(chunks)
