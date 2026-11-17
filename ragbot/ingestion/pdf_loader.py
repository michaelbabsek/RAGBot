from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PdfPage:
    number: int  # 1-basiert
    text: str
    image_path: str


def load_pdf(pdf_path: str | Path, image_dir: str | Path, dpi: int = 150) -> list[PdfPage]:
    """Extrahiert pro Seite den Text und rendert die ganze Seite als PNG."""
    pdf_path = Path(pdf_path)
    out_dir = Path(image_dir) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            number = page.number + 1
            image_path = out_dir / f'page-{number:04d}.png'
            page.get_pixmap(dpi=dpi).save(image_path)
            pages.append(PdfPage(number=number, text=page.get_text(), image_path=str(image_path)))
    return pages
