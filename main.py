import argparse

from ragbot import ChatSession, Settings
from ragbot.embeddings import OllamaEmbedder
from ragbot.ingestion import ImageDescriber, IngestionPipeline, TextChunker
from ragbot.llm import OllamaLLM
from ragbot.retriever import Retriever
from ragbot.vectorstore import VectorStore


def ingest(settings: Settings, pdf_paths: list[str]) -> None:
    pipeline = IngestionPipeline(
        store=VectorStore(settings.chroma_path, settings.collection_name),
        embedder=OllamaEmbedder(settings.embed_model),
        chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
        describer=ImageDescriber(OllamaLLM(settings.chat_model)),
        image_dir=settings.page_image_dir,
        render_dpi=settings.render_dpi,
    )
    for path in pdf_paths:
        print(f'Verarbeite {path} ...')
        count = pipeline.ingest_pdf(path)
        print(f'  {count} Chunks gespeichert.')


def chat(settings: Settings) -> None:
    store = VectorStore(settings.chroma_path, settings.collection_name)
    retriever = Retriever(store, OllamaEmbedder(settings.embed_model), top_k=settings.top_k)
    session = ChatSession(
        llm=OllamaLLM(settings.chat_model),
        retriever=retriever,
        max_history_turns=settings.max_history_turns,
    )

    print('Befehle: /reset, /exit')
    while True:
        try:
            user_input = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input == '/exit':
            break
        if user_input == '/reset':
            session.reset()
            print('Verlauf gelöscht.')
            continue

        sources, tokens = session.ask_stream(user_input)
        for token in tokens:
            print(token, end='', flush=True)
        print()
        if sources:
            print('Quellen: ' + ', '.join(f'{s.chunk.id} ({s.distance:.2f})' for s in sources))


def main() -> None:
    parser = argparse.ArgumentParser(description='RAGBot')
    sub = parser.add_subparsers(dest='command')
    ingest_parser = sub.add_parser('ingest', help='PDFs einlesen, chunken und embedden')
    ingest_parser.add_argument('pdfs', nargs='+')
    sub.add_parser('chat', help='Chat starten (Standard)')
    args = parser.parse_args()

    settings = Settings()
    if args.command == 'ingest':
        ingest(settings, args.pdfs)
    else:
        chat(settings)


if __name__ == '__main__':
    main()
