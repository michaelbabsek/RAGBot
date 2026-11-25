"""Stufe 4: Auswertung der Läufe und Report.

Metriken pro Konfiguration (gesamt / text / visual):
  Recall@4  – mindestens eines der 4 weitergegebenen Items gehört zur Gold-Seite
  Recall@16 – mindestens einer der 16 Kandidaten gehört zur Gold-Seite (Obergrenze für das Reranking)
  MRR@16    – 1 / Rang des ersten Items der Gold-Seite in der finalen Reihenfolge
"Gold-Seite" = Seite inkl. inhaltsgleicher Kopien (Äquivalenzklasse); zusätzlich strikt (nur exakte Seite).
95-%-KI per Bootstrap (Perzentil), gepaarte Vergleiche mit exaktem McNemar-Test.

Ausgaben: eval/results/report.md, per_question.csv, metrics.json
"""
import argparse
import csv
import json

import numpy as np
from scipy.stats import binomtest

from common import PRELIM_HELP, load_config, read_jsonl, results_dir, run_metadata, runs_dir, write_json
from run_retrieval import load_testset

MIN0 = 'MM-R (min_score=0)'
SUBSETS = ('all', 'text', 'visual')


def question_metrics(q: dict, rec: dict, top_n: int, gold: set[str], min_score: float | None = None) -> dict:
    page_of = {c['id']: c['page_key'] for c in rec['candidates']}
    ranking = rec['ranking']
    if min_score is not None:
        score = {c['id']: c['rerank_score'] for c in rec['candidates']}
        passed = [cid for cid in ranking if score[cid] >= min_score][:top_n]
    else:
        passed = ranking[:top_n]
    first = next((i for i, cid in enumerate(ranking, 1) if page_of[cid] in gold), None)
    return {
        'hit4': int(any(page_of[cid] in gold for cid in passed)),
        'hit16': int(first is not None),
        'rr': 1.0 / first if first else 0.0,
        'n_passed': len(passed),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, b: int, alpha: float) -> tuple[float, float]:
    if len(values) == 0:
        return float('nan'), float('nan')
    idx = rng.integers(0, len(values), size=(b, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Gepaarter exakter McNemar-Test (Binomialtest auf den diskordanten Paaren), a = Basis, b = neu."""
    improved = int(((a == 0) & (b == 1)).sum())
    worsened = int(((a == 1) & (b == 0)).sum())
    n_disc = improved + worsened
    p = binomtest(improved, n_disc, 0.5).pvalue if n_disc else 1.0
    return {'improved': improved, 'worsened': worsened, 'p_value': float(p)}


def pct(x: float) -> str:
    return f'{100 * x:.1f} %'


def ci_str(ci: tuple[float, float]) -> str:
    return f'[{100 * ci[0]:.1f}; {100 * ci[1]:.1f}]'


def fmt_p(p: float) -> str:
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, help='Smoke-Test: wertet die Smoke-Läufe aus')
    parser.add_argument('--prelim', action='store_true', help=PRELIM_HELP)
    args = parser.parse_args()
    cfg = load_config()
    rcfg = cfg['report']
    top_n = cfg['retrieval']['top_n']
    testset_path, questions = load_testset(cfg, args.limit, args.prelim)
    qs = {q['id']: q for q in questions}

    runs = {}
    for name in cfg['configs']:
        recs = {r['id']: r for r in read_jsonl(runs_dir(cfg, args.limit, args.prelim) / f'{name}.jsonl')}
        missing = [qid for qid in qs if qid not in recs]
        if missing:
            raise SystemExit(f'{name}: {len(missing)} Fragen fehlen im Lauf – zuerst run_retrieval.py abschließen.')
        runs[name] = recs
    names = [*cfg['configs'], MIN0]

    # Metriken pro Frage, lenient (Äquivalenzklasse) und strikt (nur exakte Seite)
    per_q = {}
    for qid, q in qs.items():
        lenient, strict = set(q['gold_pages']), {f"{q['doc_id']}#{q['page']}"}
        row = {}
        for name in names:
            rec = runs['MM-R' if name == MIN0 else name][qid]
            ms = cfg['retrieval']['rerank_min_score'] if name == MIN0 else None
            row[name] = question_metrics(q, rec, top_n, lenient, ms)
            row[name]['strict_hit4'] = question_metrics(q, rec, top_n, strict, ms)['hit4']
        per_q[qid] = row

    ids = {s: [qid for qid in qs if s == 'all' or qs[qid]['type'] == s] for s in SUBSETS}
    arr = lambda name, metric, s: np.array([per_q[qid][name][metric] for qid in ids[s]], dtype=float)

    metrics = {}
    for name in names:
        metrics[name] = {}
        for s in SUBSETS:
            m = {'n': len(ids[s])}
            for metric in ('hit4', 'strict_hit4', 'hit16', 'rr'):
                v = arr(name, metric, s)
                rng = np.random.default_rng(cfg['seed'])
                m[metric] = float(v.mean()) if len(v) else float('nan')
                m[metric + '_ci'] = bootstrap_ci(v, rng, rcfg['bootstrap_resamples'], rcfg['alpha'])
            m['avg_passed'] = float(arr(name, 'n_passed', s).mean()) if ids[s] else float('nan')
            metrics[name][s] = m

    def compare(base: str, new: str, subset: str) -> dict:
        a, b = arr(base, 'hit4', subset), arr(new, 'hit4', subset)
        res = mcnemar(a, b)
        rng = np.random.default_rng(cfg['seed'])
        res['delta'] = float(b.mean() - a.mean()) if len(a) else float('nan')
        res['delta_ci'] = bootstrap_ci(b - a, rng, rcfg['bootstrap_resamples'], rcfg['alpha'])
        res['n'] = len(a)
        return res

    comparisons = {
        'rerank_all': compare('MM-0', 'MM-R', 'all'),
        'rerank_text': compare('MM-0', 'MM-R', 'text'),
        'rerank_visual': compare('MM-0', 'MM-R', 'visual'),
        'images_visual': compare('TXT-R', 'MM-R', 'visual'),
        'images_text': compare('TXT-R', 'MM-R', 'text'),
        'images_visual_no_rerank': compare('TXT-0', 'MM-0', 'visual'),
        'rerank_txt_all': compare('TXT-0', 'TXT-R', 'all'),
    }

    latency = {}
    for name in cfg['configs']:
        ret = np.array([runs[name][qid]['latency_retrieval_ms'] for qid in qs])
        rer = np.array([runs[name][qid]['latency_rerank_ms'] or 0.0 for qid in qs])
        latency[name] = {
            k: {'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95))}
            for k, v in (('retrieval', ret), ('rerank', rer), ('total', ret + rer))
        }

    out = results_dir(cfg, args.limit, args.prelim)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'per_question.csv', qs, per_q, names, runs)
    write_json(out / 'metrics.json', {
        'meta': run_metadata(testset=str(testset_path), run_meta={
            n: json.loads((runs_dir(cfg, args.limit, args.prelim) / f'{n}.meta.json').read_text()) for n in cfg['configs']
        }),
        'metrics': metrics, 'comparisons': comparisons, 'latency': latency,
    })
    report = render(cfg, qs, per_q, metrics, comparisons, latency, runs, ids)
    if args.prelim:
        report = report.replace('\n', '\n\n> ⚠️ **VORLÄUFIG** – Testset mit ungeprüften keep-Vorschlägen, keine manuelle Durchsicht. '
                                'Zahlen nicht für den Lebenslauf verwenden.\n', 1)
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report.split('## 2.')[0])
    print(f'-> {out / "report.md"}')


def write_csv(path, qs, per_q, names, runs) -> None:
    fields = ['id', 'type', 'subtype', 'doc_id', 'page', 'n_gold_pages', 'question']
    for n in names:
        fields += [f'{n}:hit4', f'{n}:strict_hit4', f'{n}:hit16', f'{n}:rr', f'{n}:n_passed']
    for n in runs:
        fields += [f'{n}:latency_retrieval_ms', f'{n}:latency_rerank_ms']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for qid, q in qs.items():
            row = {'id': qid, 'type': q['type'], 'subtype': q['subtype'], 'doc_id': q['doc_id'],
                   'page': q['page'], 'n_gold_pages': len(q['gold_pages']), 'question': q['question']}
            for n in names:
                for k in ('hit4', 'strict_hit4', 'hit16', 'rr', 'n_passed'):
                    row[f'{n}:{k}'] = round(per_q[qid][n][k], 4)
            for n, recs in runs.items():
                row[f'{n}:latency_retrieval_ms'] = recs[qid]['latency_retrieval_ms']
                row[f'{n}:latency_rerank_ms'] = recs[qid]['latency_rerank_ms']
            w.writerow(row)


def verb(delta: float, up: str, down: str) -> str:
    return up if delta > 0 else down if delta < 0 else 'left unchanged'


def render(cfg, qs, per_q, metrics, comparisons, latency, runs, ids) -> str:
    alpha = cfg['report']['alpha']
    n, n_vis, n_txt = len(ids['all']), len(ids['visual']), len(ids['text'])
    X, Y = metrics['MM-0']['all'], metrics['MM-R']['all']
    A, B = metrics['TXT-R']['visual'], metrics['MM-R']['visual']
    cr, ci = comparisons['rerank_all'], comparisons['images_visual']
    L = []
    L.append('# Retrieval-Evaluation RAGBot\n')
    L.append(f"Testset: N = {n} generierte Fragen (text {n_txt}, visual {n_vis}). "
             f"Treffer = Gold-Seite inkl. inhaltsgleicher Kopien; 95-%-KI per Bootstrap "
             f"({cfg['report']['bootstrap_resamples']} Resamples, Seed {cfg['seed']}); "
             f"p-Werte: exakter McNemar-Test (zweiseitig).\n")

    L.append('## 1. CV-Zahlen\n')
    L.append('| Größe | Wert | 95-%-KI | Basis |')
    L.append('|---|---|---|---|')
    L.append(f'| N | {n} | – | alle Fragen |')
    L.append(f'| n_visual | {n_vis} | – | visuelle Fragen |')
    L.append(f"| X = Recall@4 MM-0 (ohne Reranking) | {pct(X['hit4'])} | {ci_str(X['hit4_ci'])} | alle |")
    L.append(f"| Y = Recall@4 MM-R (mit Reranking) | {pct(Y['hit4'])} | {ci_str(Y['hit4_ci'])} | alle |")
    L.append(f"| A = Recall@4 TXT-R (nur Text) | {pct(A['hit4'])} | {ci_str(A['hit4_ci'])} | visual |")
    L.append(f"| B = Recall@4 MM-R (Text + Seitenbilder) | {pct(B['hit4'])} | {ci_str(B['hit4_ci'])} | visual |")
    L.append('')
    for label, c in (('Reranking (MM-0 → MM-R, alle Fragen)', cr), ('Seitenbilder (TXT-R → MM-R, visuelle Fragen)', ci)):
        sig = 'signifikant' if c['p_value'] < alpha else '**nicht signifikant**'
        L.append(f"- {label}: Δ = {100 * c['delta']:+.1f} Pp., 95-%-KI {ci_str(c['delta_ci'])} Pp., "
                 f"p = {fmt_p(c['p_value'])} ({sig} bei α = {alpha}); "
                 f"{c['improved']} Fragen verbessert, {c['worsened']} verschlechtert.")
    L.append('')
    line = (f"> Evaluated retrieval on {n} generated test questions: re-ranking "
            f"{verb(cr['delta'], 'raised', 'lowered')} Recall@4 from {100 * X['hit4']:.0f}% to {100 * Y['hit4']:.0f}%; "
            f"page images {verb(ci['delta'], 'lifted', 'lowered')} it from {100 * A['hit4']:.0f}% to "
            f"{100 * B['hit4']:.0f}% on visual questions compared with text-only search")
    L.append('Ausgefüllte Lebenslauf-Zeile:\n')
    L.append(line + '\n')
    warnings = []
    if cr['p_value'] >= alpha:
        warnings.append(f"Der Reranking-Effekt ist nicht signifikant (p = {fmt_p(cr['p_value'])}).")
    if ci['p_value'] >= alpha:
        warnings.append(f"Der Seitenbild-Effekt ist nicht signifikant (p = {fmt_p(ci['p_value'])}, n = {n_vis}).")
    if cr['delta'] <= 0 or ci['delta'] <= 0:
        warnings.append('Mindestens ein Effekt ist null oder negativ – die Formulierung oben gibt die Richtung wieder.')
    for w in warnings:
        L.append(f'**Hinweis:** {w}  ')
    if warnings:
        L.append('')
    min0 = metrics[MIN0]['all']
    L.append(f"Zum Vergleich das Produktivverhalten mit Score-Schwelle (min_score = "
             f"{cfg['retrieval']['rerank_min_score']}): Recall@4 = {pct(min0['hit4'])} {ci_str(min0['hit4_ci'])}, "
             f"im Mittel {min0['avg_passed']:.2f} weitergegebene Items.\n")

    L.append('## 2. Alle Konfigurationen\n')
    L.append('| Konfiguration | Teilmenge | n | Recall@4 [95-%-KI] | Recall@4 strikt | Recall@16 [95-%-KI] | MRR@16 [95-%-KI] | Ø Items |')
    L.append('|---|---|---|---|---|---|---|---|')
    for name, ms in metrics.items():
        for s in SUBSETS:
            m = ms[s]
            L.append(f"| {name} | {s} | {m['n']} | {pct(m['hit4'])} {ci_str(m['hit4_ci'])} | {pct(m['strict_hit4'])} | "
                     f"{pct(m['hit16'])} {ci_str(m['hit16_ci'])} | {m['rr']:.3f} "
                     f"[{m['rr_ci'][0]:.3f}; {m['rr_ci'][1]:.3f}] | {m['avg_passed']:.2f} |")
    L.append('')
    L.append('Gepaarte Vergleiche (Recall@4):\n')
    L.append('| Vergleich | Teilmenge | n | Δ [95-%-KI] | verbessert | verschlechtert | p (McNemar) |')
    L.append('|---|---|---|---|---|---|---|')
    labels = {
        'rerank_all': ('MM-0 → MM-R', 'all'), 'rerank_text': ('MM-0 → MM-R', 'text'),
        'rerank_visual': ('MM-0 → MM-R', 'visual'), 'rerank_txt_all': ('TXT-0 → TXT-R', 'all'),
        'images_visual': ('TXT-R → MM-R', 'visual'), 'images_text': ('TXT-R → MM-R', 'text'),
        'images_visual_no_rerank': ('TXT-0 → MM-0', 'visual'),
    }
    for key, (lab, s) in labels.items():
        c = comparisons[key]
        L.append(f"| {lab} | {s} | {c['n']} | {100 * c['delta']:+.1f} Pp. {ci_str(c['delta_ci'])} | "
                 f"{c['improved']} | {c['worsened']} | {fmt_p(c['p_value'])} |")
    L.append('')
    no_text = sum(1 for qid in ids['visual'] if qs[qid]['subtype'] == 'scan')
    if no_text:
        L.append(f'Hinweis: {no_text} visuelle Fragen stammen von Scans ohne Text-Chunk; '
                 'für TXT-* sind sie per Konstruktion nicht auffindbar.\n')

    L.append('Latenz pro Anfrage (ms):\n')
    L.append('| Konfiguration | Retrieval p50 | Retrieval p95 | Reranking p50 | Reranking p95 | Gesamt p50 | Gesamt p95 |')
    L.append('|---|---|---|---|---|---|---|')
    for name, lat in latency.items():
        L.append(f"| {name} | {lat['retrieval']['p50']:.0f} | {lat['retrieval']['p95']:.0f} | "
                 f"{lat['rerank']['p50']:.0f} | {lat['rerank']['p95']:.0f} | "
                 f"{lat['total']['p50']:.0f} | {lat['total']['p95']:.0f} |")
    L.append('')

    L.append('## 3. Beispiele\n')
    L += examples(cfg, qs, per_q, runs, ids)
    return '\n'.join(L) + '\n'


def examples(cfg, qs, per_q, runs, ids) -> list[str]:
    n_ex = cfg['report']['n_examples']
    top_n = cfg['retrieval']['top_n']
    rng = np.random.default_rng(cfg['seed'])

    def pick(base, new, subset, flip, k):
        cand = [qid for qid in ids[subset] if per_q[qid][base]['hit4'] == flip[0] and per_q[qid][new]['hit4'] == flip[1]]
        return list(rng.permutation(cand)[:k]) if cand else []

    groups = [
        ('Reranking hat geholfen', 'MM-0', 'MM-R', pick('MM-0', 'MM-R', 'all', (0, 1), n_ex // 4 + 1)),
        ('Reranking hat geschadet', 'MM-0', 'MM-R', pick('MM-0', 'MM-R', 'all', (1, 0), max(1, n_ex // 8))),
        ('Seitenbilder haben geholfen', 'TXT-R', 'MM-R', pick('TXT-R', 'MM-R', 'visual', (0, 1), n_ex // 4 + 1)),
        ('Seitenbilder haben geschadet', 'TXT-R', 'MM-R', pick('TXT-R', 'MM-R', 'visual', (1, 0), max(1, n_ex // 8))),
    ]
    L = []
    for title, base, new, qids in groups:
        if not qids:
            continue
        L.append(f'### {title} ({base} → {new})\n')
        for qid in qids:
            q = qs[qid]
            gold = set(q['gold_pages'])
            L.append(f"**{qid}** ({q['type']}{'/' + q['subtype'] if q['subtype'] else ''}) – {q['question']}  ")
            L.append(f"Gold-Seite: {q['doc_id']} S. {q['page']}"
                     + (f" (+{len(gold) - 1} inhaltsgleiche)" if len(gold) > 1 else '') + '\n')
            for label, name in (('vorher', base), ('nachher', new)):
                rec = runs[name][qid]
                cands = {c['id']: c for c in rec['candidates']}
                items = []
                for cid in rec['top_n'][:top_n]:
                    c = cands[cid]
                    mark = '✅' if c['page_key'] in gold else '·'
                    kind = 'Bild' if c['modality'] == 'image' else 'Text'
                    items.append(f"{mark} {c['page_key'].replace('#', ' S. ')} ({kind})")
                L.append(f"- {label} ({name}): " + '; '.join(items))
            L.append('')
    return L


if __name__ == '__main__':
    main()
