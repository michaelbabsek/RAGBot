"""Stufe 0: Kopiert den RAGBot-Index nach eval/index_snapshot/ und hält einen Inhalts-Hash fest.

Alle späteren Läufe lesen nur aus der Kopie und prüfen den Hash. Der Hash geht über den
Collection-Inhalt (IDs, Dokumente, Metadaten, Embeddings) und die Seitenbilder, nicht über die
SQLite-Datei, weil Chroma beim Öffnen intern schreiben kann.
"""
import argparse
import hashlib
import json
import shutil

import numpy as np

from common import ROOT, load_config, run_metadata, sha256_file, write_json

MANIFEST = 'manifest.json'


def collection_hash(chroma_path: str, name: str = 'docs') -> tuple[str, int]:
    import chromadb
    col = chromadb.PersistentClient(path=chroma_path).get_collection(name)
    r = col.get(include=['documents', 'metadatas', 'embeddings'])
    order = np.argsort(r['ids'])
    h = hashlib.sha256()
    for i in order:
        h.update(r['ids'][i].encode())
        h.update((r['documents'][i] or '').encode())
        h.update(json.dumps(r['metadatas'][i], sort_keys=True, ensure_ascii=False).encode())
        h.update(np.asarray(r['embeddings'][i], dtype=np.float32).tobytes())
    return h.hexdigest(), len(r['ids'])


def page_images_hash(image_dir) -> tuple[str, int]:
    h = hashlib.sha256()
    files = sorted(p for p in (ROOT / image_dir).rglob('*.png'))
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest(), len(files)


def verify(cfg: dict) -> dict:
    """Bricht ab, wenn Snapshot oder Seitenbilder vom eingefrorenen Stand abweichen."""
    snap = ROOT / cfg['paths']['index_snapshot']
    manifest_path = snap.parent / MANIFEST
    if not manifest_path.exists():
        raise SystemExit('Kein Index-Snapshot. Zuerst `python eval/freeze_index.py` ausführen.')
    manifest = json.loads(manifest_path.read_text())
    coll, _ = collection_hash(str(snap))
    imgs, _ = page_images_hash(cfg['paths']['page_image_dir'])
    if coll != manifest['collection_sha256'] or imgs != manifest['page_images_sha256']:
        raise SystemExit('Index-Snapshot oder Seitenbilder weichen vom eingefrorenen Stand ab.')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='nur Snapshot prüfen')
    args = parser.parse_args()
    cfg = load_config()
    snap = ROOT / cfg['paths']['index_snapshot']

    if args.verify:
        m = verify(cfg)
        print(f"OK: {m['n_items']} Items, {m['n_page_images']} Seitenbilder, eingefroren am {m['date']}")
        return
    if snap.exists():
        raise SystemExit(f'{snap} existiert bereits – der Index ist eingefroren. Prüfen mit --verify.')

    src = ROOT / cfg['paths']['source_index']
    src_hash, _ = collection_hash(str(src))
    shutil.copytree(src, snap)
    coll, n_items = collection_hash(str(snap))
    assert coll == src_hash, 'Kopie weicht vom Original ab'
    imgs, n_imgs = page_images_hash(cfg['paths']['page_image_dir'])
    manifest = run_metadata(
        source_index=str(src.relative_to(ROOT)),
        collection_sha256=coll,
        n_items=n_items,
        page_images_sha256=imgs,
        n_page_images=n_imgs,
    )
    write_json(snap.parent / MANIFEST, manifest)
    print(f'Index eingefroren: {n_items} Items, {n_imgs} Seitenbilder -> {snap}')


if __name__ == '__main__':
    main()
