import ollama
from ollama import chat, ChatResponse
import chromadb

EMBED_MODEL = 'embeddinggemma-2'
TEXT_MODEL = 'gemma4:26b'

def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings

client = chromadb.PersistentClient(path='./chroma_db')

collection = client.get_or_create_collection(
    name='docs',
    metadata={'hnsw:space': 'cosine'},
)

chunks = [
    'Die Klausur in Datenbanksysteme ist am 12. Februar.',
    'SQL-Joins verbinden Tabellen über gemeinsame Schlüssel.',
    'Normalformen reduzieren Redundanz in relationalen Schemata.',
]

collection.add(
    ids=[f'chunk-{i}' for i in range(len(chunks))],
    documents=chunks,                 # Originaltext, den du später ans LLM gibst
    embeddings=embed(chunks),         # deine eigenen Vektoren
    metadatas=[{'source': 'skript.pdf'} for _ in chunks],
)

def retrieve(query: str, k: int = 4) -> list[str]:
    result = collection.query(
        query_embeddings=embed([query]),  # wichtig: query_embeddings, nicht query_texts
        n_results=k,
    )
    return result['documents'][0]

messages = []

while True:
    user_input = input('> ')
    retrieved = [str(s) for s in retrieve(user_input)]

    context = '\n\n---\n\n'.join(retrieved)  # Separator nur zwischen den Chunks
    content = f'Kontext:\n{context}\n\nFrage: {user_input}'

    response = chat(
        model='gemma4:26b',
        messages=messages + [{'role': 'user', 'content': content}],
    )

    messages.append({'role': 'user', 'content': user_input})
    print(response.message.content)

    messages.append({'role': 'assistant', 'content': response.message.content})

