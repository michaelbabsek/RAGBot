"""Gemeinsame Hilfsfunktionen für alle Eval-Stufen."""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Chunk-Metadaten enthalten relative Bildpfade (./data/pages/...), daher immer vom Repo-Root aus arbeiten
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGES = [
    'torch', 'transformers', 'sentence-transformers', 'chromadb', 'pymupdf', 'ollama',
    'pyyaml', 'numpy', 'scipy', 'pillow', 'torchcodec', 'accelerate', 'qwen-vl-utils',
]


def load_config() -> dict[str, Any]:
    with open(ROOT / 'eval' / 'config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def nfc(s: str) -> str:
    # macOS liefert Dateinamen in NFD, im Index steht NFC
    return unicodedata.normalize('NFC', s)


def data_dir(cfg: dict, limit: int | None, prelim: bool = False) -> Path:
    """Smoke-Läufe (--limit) und vorläufige Läufe (--prelim) schreiben in eigene Unterverzeichnisse."""
    d = ROOT / cfg['paths']['data_dir']
    return d / 'smoke' if limit else d / 'prelim' if prelim else d


def runs_dir(cfg: dict, limit: int | None, prelim: bool = False) -> Path:
    d = ROOT / cfg['paths']['runs_dir']
    return d / 'smoke' if limit else d / 'prelim' if prelim else d


def results_dir(cfg: dict, limit: int | None, prelim: bool = False) -> Path:
    d = ROOT / cfg['paths']['results_dir']
    return d / 'smoke' if limit else d / 'prelim' if prelim else d


PRELIM_HELP = ('vorläufiger Lauf mit den ungeprüften keep-Vorschlägen; eigenes Verzeichnis prelim/, '
               'blockiert das echte Einfrieren nicht')


def page_key(doc_id: str, page: int) -> str:
    return f'{doc_id}#{page}'


def git_commit() -> str:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.call(
            ['git', 'diff', '--quiet', 'HEAD', '--', ':(glob)ragbot/**/*.py', 'main.py']
        ) != 0
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def package_versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = 'missing'
    return out


def ollama_version() -> str:
    try:
        return subprocess.check_output(['ollama', '--version'], text=True).strip()
    except Exception:
        return 'unknown'


def run_metadata(**extra: Any) -> dict[str, Any]:
    return {
        'date': dt.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': package_versions(),
        **extra,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
