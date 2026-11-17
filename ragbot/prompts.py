from ragbot.models import Modality, RetrievedChunk

SYSTEM_PROMPT = (
    'Du bist ein hilfreicher Assistent. Beantworte Fragen auf Basis des bereitgestellten Kontexts. '
    'Der Kontext kann Text und Bilder enthalten. Wenn der Kontext die Antwort nicht enthält, sag das ehrlich.'
)

SEPARATOR = '\n\n---\n\n'


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(chunks, start=1):
        source = rc.chunk.metadata.get('source', 'unbekannt')
        if 'page' in rc.chunk.metadata:
            source = f"{source}, Seite {rc.chunk.metadata['page']}"
        if rc.chunk.modality is Modality.IMAGE:
            parts.append(f'[{i}] (Bild: {source}, siehe angehängtes Bild) {rc.chunk.content}'.rstrip())
        else:
            parts.append(f'[{i}] (Quelle: {source})\n{rc.chunk.content}')
    return SEPARATOR.join(parts)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        # Explizit sagen, dass nichts gefunden wurde, damit das Modell nicht aus eigenem Wissen antwortet
        return f'Kontext: (keine relevanten Dokumente gefunden)\n\nFrage: {question}'
    return f'Kontext:\n{format_context(chunks)}\n\nFrage: {question}'

