"""Stufe 3: Retrieval-Lauf für eine Konfiguration (MM-0, MM-R, TXT-0, TXT-R) über das eingefrorene Testset.

Nutzt dieselben Komponenten wie RAGBot (HFEmbedder, VectorStore, HFReranker), aber direkt statt über
Retriever, damit alle 16 Kandidaten, die volle Reranking-Reihenfolge und die Latenzen getrennt
gespeichert werden können. Reranking läuft ohne Score-Schwelle über alle 16 Kandidaten; die
Schwelle des Produktivsystems (min_score) wird in report.py aus den gespeicherten Scores angewendet.

Ausgabe: eval/runs/<config>.jsonl (resumable) + <config>.meta.json
"""
import argparse
import json
import time

from common import (PRELIM_HELP, ROOT, append_jsonl, data_dir, load_config, page_key, read_jsonl, run_metadata,
                    runs_dir, sha256_file, write_json)
from freeze_index import verify


def load_testset(cfg: dict, limit: int | None, prelim: bool = False):
    """Mit --limit: das echte Testset, falls schon eingefroren, sonst das Smoke-Testset."""
    path = data_dir(cfg, None, prelim) / 'testset.json'
    if limit and not path.exists():
        path = data_dir(cfg, limit) / 'testset.json'
    if not path.exists():
        raise SystemExit(f'{path} fehlt – zuerst freeze_testset.py ausführen.')
    questions = json.loads(path.read_text(encoding='utf-8'))['questions']
    return path, (questions[:limit] if limit else questions)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, choices=[*cfg['configs'], 'all'])
    parser.add_argument('--limit', type=int, help='Smoke-Test: nur die ersten N Fragen (eigenes Verzeichnis)')
    parser.add_argument('--prelim', action='store_true', help=PRELIM_HELP)
    args = parser.parse_args()

    manifest = verify(cfg)
    testset_path, questions = load_testset(cfg, args.limit, args.prelim)
    testset_sha = sha256_file(testset_path)
    names = list(cfg['configs']) if args.config == 'all' else [args.config]

    from ragbot.config import Settings
    from ragbot.embeddings import HFEmbedder, default_device
    from ragbot.vectorstore import VectorStore

    settings = Settings()
    r = cfg['retrieval']
    assert r['candidates'] == settings.top_k and r['top_n'] == settings.rerank_top_n, \
        'eval/config.yaml weicht von ragbot.config.Settings ab'
    store = VectorStore(str(ROOT / cfg['paths']['index_snapshot']), settings.collection_name)
    embedder = HFEmbedder(settings.embed_model)
    reranker = None
    if any(cfg['configs'][n]['rerank'] for n in names):
        from ragbot.reranker import HFReranker
        reranker = HFReranker(settings.rerank_model)

    # Aufwärmen, damit Modell-Initialisierung auf MPS nicht in die Latenz der ersten Frage fällt
    for i in range(r['warmup_queries']):
        cands = store.query(embedder.embed_query(f'Aufwärmanfrage {i}'), k=r['candidates'])
        if reranker:
            reranker.rerank(f'Aufwärmanfrage {i}', cands, top_n=len(cands))

    out = runs_dir(cfg, args.limit, args.prelim)
    for name in names:
        conf = cfg['configs'][name]
        run_path = out / f'{name}.jsonl'
        meta_path = out / f'{name}.meta.json'
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old['testset_sha256'] != testset_sha:
                raise SystemExit(f'{run_path} gehört zu einem anderen Testset – Datei löschen oder prüfen.')
        else:
            write_json(meta_path, run_metadata(
                config=name, where=conf['where'], rerank=conf['rerank'],
                embed_model=settings.embed_model, rerank_model=settings.rerank_model if conf['rerank'] else None,
                device=default_device(), candidates=r['candidates'], top_n=r['top_n'],
                testset=str(testset_path.relative_to(ROOT)), testset_sha256=testset_sha,
                index_collection_sha256=manifest['collection_sha256'],
            ))

        done = {rec['id'] for rec in read_jsonl(run_path)}
        todo = [q for q in questions if q['id'] not in done]
        print(f'{name}: {len(done)} erledigt, {len(todo)} offen')
        for n, q in enumerate(todo, 1):
            t0 = time.perf_counter()
            cands = store.query(embedder.embed_query(q['question']), k=r['candidates'], where=conf['where'])
            t1 = time.perf_counter()
            embed_rank = {rc.chunk.id: i for i, rc in enumerate(cands, 1)}
            if conf['rerank']:
                ranked = reranker.rerank(q['question'], list(cands), top_n=len(cands), min_score=None)
                t2 = time.perf_counter()
            else:
                ranked, t2 = cands, t1
            final_rank = {rc.chunk.id: i for i, rc in enumerate(ranked, 1)}

            candidates = [{
                'id': rc.chunk.id,
                'page_key': page_key(rc.chunk.metadata['source'], int(rc.chunk.metadata['page'])),
                'modality': rc.chunk.modality.value,
                'distance': round(rc.distance, 6),
                'embed_rank': embed_rank[rc.chunk.id],
                'rerank_score': None if rc.score is None else round(rc.score, 4),
                'final_rank': final_rank[rc.chunk.id],
            } for rc in cands]
            ranking = [rc.chunk.id for rc in ranked]
            append_jsonl(run_path, {
                'id': q['id'],
                'config': name,
                'candidates': candidates,
                'ranking': ranking,
                'top_n': ranking[:r['top_n']],
                'latency_retrieval_ms': round((t1 - t0) * 1000, 1),
                'latency_rerank_ms': round((t2 - t1) * 1000, 1) if conf['rerank'] else None,
            })
            gold = set(q['gold_pages'])
            hit = any(c['page_key'] in gold for c in candidates if c['id'] in ranking[:r['top_n']])
            print(f"  [{n}/{len(todo)}] {q['id']} {q['type']:6} hit@{r['top_n']}={int(hit)} "
                  f"{(t1 - t0) * 1000:6.0f} ms + {(t2 - t1) * 1000:6.0f} ms")

        if args.limit and conf['rerank'] and conf['where'] is None:
            check_equivalence(cfg, settings, store, embedder, reranker, questions, read_jsonl(run_path))


def check_equivalence(cfg, settings, store, embedder, reranker, questions, records) -> None:
    """Smoke-Test: MM-R muss (mit min_score) genau das liefern, was ragbot.Retriever liefert."""
    from ragbot.retriever import Retriever
    retriever = Retriever(store, embedder, top_k=settings.top_k, reranker=reranker,
                          rerank_top_n=settings.rerank_top_n, rerank_min_score=settings.rerank_min_score)
    by_id = {rec['id']: rec for rec in records}
    for q in questions:
        rec = by_id[q['id']]
        scores = {c['id']: c['rerank_score'] for c in rec['candidates']}
        ours = [cid for cid in rec['ranking'] if scores[cid] >= settings.rerank_min_score][:settings.rerank_top_n]
        theirs = [rc.chunk.id for rc in retriever.retrieve(q['question'])]
        status = 'OK' if ours == theirs else 'ABWEICHUNG'
        print(f"  Äquivalenz {q['id']}: {status} {ours if status == 'OK' else (ours, theirs)}")


if __name__ == '__main__':
    main()
