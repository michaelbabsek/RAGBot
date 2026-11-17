from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Hugging Face (Embedding + Reranking), Chat läuft über Ollama
    embed_model: str = 'Qwen/Qwen3-VL-Embedding-2B'
    rerank_model: str = 'Qwen/Qwen3-VL-Reranker-2B'
    chat_model: str = 'gemma4:e4b'
    chat_temperature: float = 0.2
    chat_think: bool = False

    chroma_path: str = './chroma_db'
    collection_name: str = 'docs'

    # Ingestion
    chunk_size: int = 250  # Wörter
    chunk_overlap: float = 0.25  # Anteil von chunk_size
    page_image_dir: str = './data/pages'
    render_dpi: int = 150

    # Kandidaten aus der Vektorsuche, davon bleiben nach dem Reranking die besten rerank_top_n
    top_k: int = 16
    rerank_top_n: int = 4
    # Kandidaten mit niedrigerem Reranker-Score werden verworfen (Score ist ein Logit, < 0 = eher irrelevant)
    rerank_min_score: float = 0.0
    # Anzahl der vergangenen User/Assistant-Paare, die ans LLM gehen
    max_history_turns: int = 10
