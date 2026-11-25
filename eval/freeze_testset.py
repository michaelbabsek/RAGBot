"""Friert das Testset ein: übernimmt nur Zeilen mit keep=1 aus questions_draft.csv nach testset.json.

Prüft, dass jede Gold-Seite (inkl. Äquivalenzklasse) im eingefrorenen Index existiert.
Danach ist das Testset fix – ein erneutes Einfrieren wird verweigert.
"""
import argparse
import csv
import json
from collections import Counter

import chromadb

from common import PRELIM_HELP, ROOT, data_dir, load_config, page_key, sha256_file, write_json
from freeze_index import verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Smoke-Test: arbeitet in eval/data/smoke/')
    parser.add_argument('--prelim', action='store_true', help=PRELIM_HELP)
    args = parser.parse_args()
    cfg = load_config()
    out = data_dir(cfg, args.limit)
    target = data_dir(cfg, args.limit, args.prelim) / 'testset.json'
    if target.exists() and not args.limit and not args.prelim:
        raise SystemExit(f'{target} existiert bereits – das Testset ist eingefroren.')

    manifest = verify(cfg)
    draft = out / 'questions_draft.csv'
    with open(draft, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    errors = []
    bad_keep = [r['id'] for r in rows if r['keep'].strip() not in ('0', '1')]
    if bad_keep:
        errors.append(f'keep muss 0 oder 1 sein: {bad_keep}')
    ids = Counter(r['id'] for r in rows)
    if dupes := [i for i, c in ids.items() if c > 1]:
        errors.append(f'doppelte IDs: {dupes}')
    kept = [r for r in rows if r['keep'].strip() == '1']
    if args.limit:
        kept = kept[:args.limit]

    col = chromadb.PersistentClient(path=str(ROOT / cfg['paths']['index_snapshot'])).get_collection('docs')
    indexed = {page_key(m['source'], int(m['page'])) for m in col.get(include=['metadatas'])['metadatas']}

    questions = []
    for r in kept:
        gold_pages = json.loads(r['gold_pages'])
        own = page_key(r['doc_id'], int(r['page']))
        if own not in gold_pages:
            errors.append(f"{r['id']}: Gold-Seite {own} fehlt in gold_pages")
        if missing := [g for g in gold_pages if g not in indexed]:
            errors.append(f"{r['id']}: nicht im Index: {missing}")
        if not r['question'].strip() or not r['answer'].strip():
            errors.append(f"{r['id']}: Frage oder Antwort leer")
        if r['type'] not in ('text', 'visual'):
            errors.append(f"{r['id']}: unbekannter Typ {r['type']}")
        if not (ROOT / r['page_image_path']).exists():
            errors.append(f"{r['id']}: Seitenbild fehlt")
        questions.append({
            'id': r['id'],
            'question': r['question'].strip(),
            'answer': r['answer'].strip(),
            'doc_id': r['doc_id'],
            'page': int(r['page']),
            'type': r['type'],
            'subtype': r['subtype'],
            'gold_pages': gold_pages,
            'page_image_path': r['page_image_path'],
        })
    if errors:
        raise SystemExit('Testset NICHT eingefroren:\n  ' + '\n  '.join(errors))

    by_type = Counter(q['type'] for q in questions)
    write_json(target, {
        'meta': {
            'draft_sha256': sha256_file(draft),
            'index_collection_sha256': manifest['collection_sha256'],
            'n': len(questions),
            'n_by_type': dict(by_type),
            'n_by_subtype': dict(Counter(f"{q['type']}/{q['subtype'] or '-'}" for q in questions)),
            'n_docs': len({q['doc_id'] for q in questions}),
        },
        'questions': questions,
    })
    print(f'Testset eingefroren: N = {len(questions)} '
          f"(text {by_type['text']}, visual {by_type['visual']}) aus "
          f"{len({q['doc_id'] for q in questions})} Dokumenten -> {target}")
    print(f'Verworfen (keep=0): {len(rows) - sum(r["keep"].strip() == "1" for r in rows)}')


if __name__ == '__main__':
    main()
