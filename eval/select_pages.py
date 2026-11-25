"""Stufe 1: Analysiert alle Seiten der indexierten PDFs und zieht stratifiziert Seiten für die Fragen.

Ausgaben (in eval/data/, mit --limit in eval/data/smoke/):
  pages.csv           alle Seiten mit Merkmalen, Klasse, Ausschlussgrund und Äquivalenzklasse
  selected_pages.csv  gezogene Seiten (eine pro Äquivalenzklasse), Eingabe für Stufe 2

Äquivalenzklassen fassen inhaltlich (fast) gleiche Seiten zusammen (kopierte Foliensätze,
schrittweise aufgebaute Folien, Blatt/Lösung). Ein Treffer auf eine Seite der Klasse zählt in
der Auswertung als Treffer der Gold-Seite (zusätzlich wird strikt ausgewertet).

Optional: eval/data/page_overrides.csv (doc_id,page,page_class) überschreibt die Klassifikation.
"""
import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import chromadb
import pymupdf
from PIL import Image

from common import ROOT, data_dir, load_config, nfc, page_key

pymupdf.TOOLS.mupdf_display_errors(False)

FIELDS = [
    'doc_id', 'page', 'page_class', 'subtype', 'exclude_reason', 'chars', 'words',
    'n_images', 'max_image_frac', 'n_drawings', 'drawings_over_median', 'has_text_chunk',
    'eq_class', 'eq_size', 'image_path',
]
WORD_RE = re.compile(r'\w{4,}')


def index_pages(snapshot: str) -> tuple[dict[tuple[str, int], str], set[tuple[str, int]]]:
    """(doc_id, page) -> Bildpfad sowie die Menge der Seiten mit Text-Chunk, direkt aus dem Snapshot."""
    col = chromadb.PersistentClient(path=snapshot).get_collection('docs')
    metas = col.get(include=['metadatas'])['metadatas']
    images, texts = {}, set()
    for m in metas:
        key = (m['source'], int(m['page']))
        if m['modality'] == 'image':
            images[key] = m['image_path']
        else:
            texts.add(key)
    return images, texts


def dhash(path: str, size: int = 16) -> int:
    img = Image.open(path).convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(img.get_flattened_data())
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def analyze(cfg: dict) -> list[dict]:
    sel = cfg['select']
    images, texts = index_pages(cfg['paths']['index_snapshot'])
    rows = []
    for pdf in sorted((ROOT / cfg['paths']['docs_dir']).glob('*.pdf')):
        doc_id = nfc(pdf.name)
        with pymupdf.open(pdf) as doc:
            drawings = [len(p.get_drawings()) for p in doc]
            median = statistics.median(drawings) if drawings else 0
            for p, n_draw in zip(doc, drawings):
                text = p.get_text()
                area = p.rect.width * p.rect.height
                fracs = []
                for info in p.get_image_info():
                    r = pymupdf.Rect(info['bbox']) & p.rect
                    fracs.append(0.0 if r.is_empty else r.width * r.height / area)
                page_no = p.number + 1
                rows.append({
                    'doc_id': doc_id,
                    'page': page_no,
                    'text': text,
                    'chars': len(text.strip()),
                    'words': len(text.split()),
                    'n_images': len(fracs),
                    'max_image_frac': round(max(fracs, default=0.0), 3),
                    'n_drawings': n_draw,
                    'drawings_over_median': n_draw - median,
                    'has_text_chunk': int((doc_id, page_no) in texts),
                    'image_path': images.get((doc_id, page_no), ''),
                })
    missing = [(r['doc_id'], r['page']) for r in rows if not r['image_path']]
    if missing:
        raise SystemExit(f'{len(missing)} Seiten fehlen im Index, z. B. {missing[:3]}')
    return rows


def classify(rows: list[dict], cfg: dict) -> None:
    sel = cfg['select']
    toc = re.compile(r'\b(' + '|'.join(map(re.escape, sel['toc_keywords'])) + r')\b', re.IGNORECASE)
    excluded_docs = {nfc(d) for d in sel['exclude_docs']}
    for r in rows:
        r['subtype'] = ''
        r['exclude_reason'] = ''
        big = r['max_image_frac']
        if r['chars'] < sel['scan_max_chars'] and big >= sel['scan_min_image_frac']:
            r['page_class'], r['subtype'] = 'visual', 'scan'
        elif r['chars'] < sel['empty_max_chars'] and big < sel['visual_min_image_frac']:
            r['page_class'] = 'other'
            r['exclude_reason'] = 'empty'
        elif big >= sel['visual_min_image_frac']:
            r['page_class'], r['subtype'] = 'visual', 'figure'
        elif r['drawings_over_median'] >= sel['visual_min_drawings_over_median']:
            r['page_class'], r['subtype'] = 'visual', 'diagram'
        elif (r['chars'] >= sel['text_min_chars'] and big <= sel['text_max_image_frac']
              and r['drawings_over_median'] <= sel['text_max_drawings_over_median']):
            r['page_class'] = 'text'
        else:
            r['page_class'] = 'other'
            r['exclude_reason'] = 'mixed'

        if r['exclude_reason']:
            continue
        if r['doc_id'] in excluded_docs:
            r['exclude_reason'] = 'excluded_doc'
        elif r['page'] == 1 and r['words'] <= sel['title_max_words'] and r['subtype'] != 'scan':
            r['exclude_reason'] = 'title'
        elif toc.search(r['text'][:80]):
            r['exclude_reason'] = 'toc'

    overrides = ROOT / cfg['paths']['data_dir'] / 'page_overrides.csv'
    if overrides.exists():
        with open(overrides, encoding='utf-8') as f:
            ov = {(nfc(o['doc_id']), int(o['page'])): o['page_class'] for o in csv.DictReader(f)}
        for r in rows:
            if (r['doc_id'], r['page']) in ov:
                r['page_class'] = ov[(r['doc_id'], r['page'])]
                r['subtype'] = r['subtype'] or 'override'
                r['exclude_reason'] = '' if r['page_class'] in ('text', 'visual') else 'override'
        print(f'{len(ov)} manuelle Klassifikationen aus {overrides.name} übernommen')


def equivalence_classes(rows: list[dict], cfg: dict) -> None:
    """Union-Find über fast gleiche Seiten: Wortmengen-Überdeckung bzw. Bild-Hash bei textarmen Seiten."""
    sel = cfg['select']
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    words = [set(w.lower() for w in WORD_RE.findall(r['text'])) for r in rows]
    wordy = [i for i, w in enumerate(words) if len(w) >= sel['near_dup_min_words']]
    for a_pos, a in enumerate(wordy):
        for b in wordy[a_pos + 1:]:
            wa, wb = words[a], words[b]
            if len(wa & wb) / min(len(wa), len(wb)) >= sel['near_dup_containment']:
                parent[find(a)] = find(b)

    sparse = [i for i, w in enumerate(words) if len(w) < sel['near_dup_min_words']]
    hashes = {i: dhash(rows[i]['image_path']) for i in sparse}
    for a_pos, a in enumerate(sparse):
        for b in sparse[a_pos + 1:]:
            if bin(hashes[a] ^ hashes[b]).count('1') <= 3:
                parent[find(a)] = find(b)

    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    for root, idx in members.items():
        first = min(idx, key=lambda i: (rows[i]['doc_id'], rows[i]['page']))
        label = page_key(rows[first]['doc_id'], rows[first]['page'])
        for i in idx:
            rows[i]['eq_class'] = label
            rows[i]['eq_size'] = len(idx)


def sample(rows: list[dict], cfg: dict) -> list[dict]:
    """Round-Robin über die Dokumente (feste Seeds), eine Seite pro Äquivalenzklasse.
    Visuelle Seiten zuerst, weil sie knapper sind."""
    rng = random.Random(cfg['seed'])
    used_classes = set()
    selected = []
    for page_type, quota in (('visual', cfg['select']['n_visual']), ('text', cfg['select']['n_text'])):
        by_doc = defaultdict(list)
        for r in rows:
            if r['page_class'] == page_type and not r['exclude_reason']:
                by_doc[r['doc_id']].append(r)
        docs = sorted(by_doc)
        rng.shuffle(docs)
        for d in docs:
            rng.shuffle(by_doc[d])
        picked = []
        while len(picked) < quota and any(by_doc.values()):
            for d in docs:
                while by_doc[d]:
                    r = by_doc[d].pop()
                    if r['eq_class'] not in used_classes:
                        used_classes.add(r['eq_class'])
                        picked.append(r)
                        break
                if len(picked) >= quota:
                    break
        if len(picked) < quota:
            print(f'WARNUNG: nur {len(picked)} von {quota} {page_type}-Seiten verfügbar')
        selected += picked

    gold = defaultdict(list)
    for r in rows:
        gold[r['eq_class']].append(page_key(r['doc_id'], r['page']))
    return [{
        'doc_id': r['doc_id'],
        'page': r['page'],
        'type': r['page_class'],
        'subtype': r['subtype'],
        'chars': r['chars'],
        'eq_class': r['eq_class'],
        'gold_pages': json.dumps(sorted(gold[r['eq_class']]), ensure_ascii=False),
        'image_path': r['image_path'],
    } for r in selected]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, help='Smoke-Test: nur so viele Seiten auswählen')
    args = parser.parse_args()
    cfg = load_config()
    out = data_dir(cfg, args.limit)

    rows = analyze(cfg)
    classify(rows, cfg)
    equivalence_classes(rows, cfg)
    selected = sample(rows, cfg)
    if args.limit:
        # abwechselnd visuell/text, damit der Smoke-Test beide Typen abdeckt
        vis = [s for s in selected if s['type'] == 'visual']
        txt = [s for s in selected if s['type'] == 'text']
        mixed = [x for pair in zip(vis, txt) for x in pair] + vis[len(txt):] + txt[len(vis):]
        selected = mixed[:args.limit]

    write_csv(out / 'pages.csv', rows, FIELDS)
    write_csv(out / 'selected_pages.csv', selected, list(selected[0]))

    n = len(rows)
    count = lambda pred: sum(1 for r in rows if pred(r))
    print(f'{n} Seiten aus {len({r["doc_id"] for r in rows})} Dokumenten analysiert')
    for c in ('text', 'visual', 'other'):
        print(f'  {c:7} {count(lambda r: r["page_class"] == c):5}   davon nutzbar: '
              f'{count(lambda r: r["page_class"] == c and not r["exclude_reason"])}')
    for st in ('figure', 'diagram', 'scan'):
        print(f'    visual/{st:8} {count(lambda r: r["subtype"] == st and not r["exclude_reason"]):4} nutzbar')
    reasons = defaultdict(int)
    for r in rows:
        reasons[r['exclude_reason'] or '-'] += 1
    print('  Ausschlussgründe:', dict(sorted(reasons.items())))
    multi = {r['eq_class'] for r in rows if r['eq_size'] > 1}
    print(f'  Äquivalenzklassen mit >1 Seite: {len(multi)} ({count(lambda r: r["eq_size"] > 1)} Seiten)')
    sel_types = defaultdict(int)
    for s in selected:
        sel_types[s['type']] += 1
    print(f'Ausgewählt: {dict(sel_types)} aus {len({s["doc_id"] for s in selected})} Dokumenten -> {out}')


if __name__ == '__main__':
    main()
