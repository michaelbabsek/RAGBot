import re


class TextChunker:
    """Teilt Text in Chunks von `chunk_size` Wörtern. Benachbarte Chunks überlappen
    sich um den Anteil `overlap` (z.B. 0.25 = 25 % von `chunk_size`).
    """

    def __init__(self, chunk_size: int = 250, overlap: float = 0.25):
        if not 0 <= overlap < 1:
            raise ValueError('overlap muss zwischen 0 (inkl.) und 1 liegen')
        self.chunk_size = chunk_size
        self.step = max(1, chunk_size - round(chunk_size * overlap))

    def split(self, text: str) -> list[str]:
        words = _normalize(text).split()
        chunks = []
        for start in range(0, len(words), self.step):
            chunks.append(' '.join(words[start:start + self.chunk_size]))
            # Letztes Fenster erreicht das Textende, weitere wären nur Overlap
            if start + self.chunk_size >= len(words):
                break
        return chunks


def _normalize(text: str) -> str:
    text = text.replace('\r\n', '\n')
    return re.sub(r'-\n(?=[a-zäöüß])', '', text)  # Silbentrennung am Zeilenende
