"""Stufe 2: Erzeugt pro ausgewählter Seite eine Frage + Kurzantwort mit einem lokalen Ollama-Modell.

- text-Seiten: Frage aus dem extrahierten Seitentext
- visual-Seiten: Frage aus dem gerenderten Seitenbild, die nur über die Abbildung beantwortbar sein soll;
  danach Filter: dasselbe Modell bekommt NUR den extrahierten Text und entscheidet, ob die Frage damit
  beantwortbar ist (ja -> keep=0). Bei Scans entfällt der Filter.

Rohantworten werden in generation_cache.jsonl gecacht (resumable). Ausgabe: questions_draft.csv
zur manuellen Prüfung (keep=0/1, Frage/Antwort ggf. umformulieren). Danach: freeze_testset.py.
"""
import argparse
import csv
import json
import re
import time

import ollama

from common import (ROOT, append_jsonl, data_dir, load_config, page_key, read_jsonl,
                    run_metadata, write_json)

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)

DRAFT_FIELDS = [
    'id', 'question', 'answer', 'doc_id', 'page', 'type', 'page_image_path', 'filter_result', 'keep',
    'subtype', 'gold_pages', 'figure', 'filter_reason', 'max_verbatim_words', 'notes',
]

COMMON_RULES = """Regeln:
- Die Frage muss für sich allein verständlich sein – wie eine Suchanfrage von jemandem, der die Seite nie gesehen hat. Nenne das konkrete Thema mit den passenden Fachbegriffen. Verboten sind vage Bezüge wie „die genannten Ansätze“, „ein bestimmtes Ziel“, „dieser Code“, „das Beispiel“ sowie Formulierungen wie „werden genannt“, „wird beschrieben“. Schlecht: „Welche zwei Vorteile werden genannt?“ Gut: „Welche Vorteile hat Scrum gegenüber dem Wasserfallmodell bei sich ändernden Anforderungen?“
- Die Frage muss sich gezielt auf den Inhalt DIESER Seite beziehen und darf nicht so allgemein sein, dass viele andere Seiten sie ebenso beantworten.
- Formuliere natürlich. Übernimm keine wörtlichen Formulierungen der Seite: höchstens 3 aufeinanderfolgende Wörter dürfen identisch sein; nutze Umschreibungen. Fachbegriffe und Eigennamen sind erlaubt.
- Keine Verweise wie „auf dieser Folie“, „im Bild“, „in der Abbildung“, „laut dem Diagramm“, „laut Text“, „im Dokument“.
- Frage und Antwort in der Sprache der Seite (meist Deutsch).
- Die Antwort ist kurz (höchstens zwei Sätze) und vollständig aus der Seite belegbar."""

TEXT_PROMPT = """Du erstellst Testfragen für die Evaluation einer Dokumentensuche über Lehrunterlagen.

Hier ist der extrahierte Text einer Seite:
<<<
{text}
>>>

Formuliere genau eine Frage, die sich mit diesem Text eindeutig beantworten lässt, und die zugehörige Antwort.

{rules}"""

VISUAL_PROMPT = """Du erstellst Testfragen für die Evaluation einer Dokumentensuche über Lehrunterlagen.

Das Bild zeigt eine Seite. Zusätzlich bekommst du den maschinell extrahierten Text dieser Seite (kann leer sein):
<<<
{text}
>>>

Formuliere genau eine Frage, die NUR mit Informationen beantwortbar ist, die ausschließlich im Bild sichtbar sind – also in einem Diagramm, Chart, einer Grafik, einem Screenshot, Foto oder gescannten Inhalt – und die sich mit dem extrahierten Text oben NICHT beantworten lässt. Geeignet sind z. B. Beziehungen oder Abläufe zwischen Elementen eines Diagramms (welcher Schritt folgt auf welchen, welche Elemente sind verbunden), Werte oder Verläufe in Charts, Beschriftungen innerhalb von Grafiken, was auf einem Foto oder Screenshot zu sehen ist.
Frage nach dem INHALT der Grafik, nicht nach ihrer Gestaltung: keine Fragen nach Farben, Markierungen, Positionen oder danach, worauf ein Pfeil oder Kasten zeigt.
Wenn die Seite keine solche visuelle Information enthält (z. B. nur Text, Tabellen oder Code, die vollständig im extrahierten Text stehen), setze "usable" auf false.
Beschreibe in "figure" in wenigen Worten, auf welches visuelle Element sich die Frage bezieht.

{rules}"""

FILTER_PROMPT = """Prüfe, ob sich eine Frage allein mit dem folgenden Text beantworten lässt.

Text:
<<<
{text}
>>>

Frage: {question}
Referenzantwort: {answer}

Lässt sich die Frage allein mit diesem Text so beantworten, dass die Referenzantwort im Wesentlichen herauskommt? Bilder, Diagramme oder Wissen außerhalb des Textes stehen NICHT zur Verfügung. Antworte mit answerable=true nur, wenn die nötige Information wirklich im Text steht."""

TEXT_SCHEMA = {
    'type': 'object',
    'properties': {'question': {'type': 'string'}, 'answer': {'type': 'string'}},
    'required': ['question', 'answer'],
}
VISUAL_SCHEMA = {
    'type': 'object',
    'properties': {
        'usable': {'type': 'boolean'},
        'figure': {'type': 'string'},
        'question': {'type': 'string'},
        'answer': {'type': 'string'},
    },
    'required': ['usable', 'figure', 'question', 'answer'],
}
FILTER_SCHEMA = {
    'type': 'object',
    'properties': {'reason': {'type': 'string'}, 'answerable': {'type': 'boolean'}},
    'required': ['reason', 'answerable'],
}

WORD_RE = re.compile(r'\w+')


def page_text(doc_id: str, page: int, docs_dir) -> str:
    # Dateinamen auf der Platte können NFD sein; Suche über NFC-Vergleich
    from common import nfc
    for pdf in docs_dir.glob('*.pdf'):
        if nfc(pdf.name) == doc_id:
            with pymupdf.open(pdf) as doc:
                return doc[page - 1].get_text()
    raise FileNotFoundError(doc_id)


def max_verbatim_words(question: str, text: str) -> int:
    q = WORD_RE.findall(question.lower())
    t = WORD_RE.findall(text.lower())
    best = 0
    for n in range(1, len(q) + 1):
        grams = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
        if any(tuple(q[i:i + n]) in grams for i in range(len(q) - n + 1)):
            best = n
        else:
            break
    return best


def _parse_json(content: str, schema: dict) -> dict:
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError(f'kein JSON in der Antwort: {content[:200]!r}')
    try:
        obj = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        # LaTeX in Formeln (\bar, \sqrt) kommt oft unescaped zurück
        # (\b, \f, \t ... in \bar, \frac, \text sind sonst gültige Escapes und würden den Text verstümmeln)
        obj = json.loads(re.sub(r'\\(?![\\"])', r'\\\\', match.group(0)), strict=False)
    py_types = {'string': str, 'boolean': bool}
    for key in schema['required']:
        value = obj.get(key)
        if schema['properties'][key]['type'] == 'boolean' and isinstance(value, str) \
                and value.strip().lower() in ('true', 'false'):
            obj[key] = value.strip().lower() == 'true'
        if not isinstance(obj.get(key), py_types[schema['properties'][key]['type']]):
            raise ValueError(f'Feld {key!r} fehlt oder hat falschen Typ: {obj}')
    return obj


class Generator:
    def __init__(self, cfg: dict):
        g = cfg['generation']
        self.model = g['model']
        self.think = g['think']
        self.options = {'temperature': g['temperature'], 'seed': g['seed']}
        self.max_chars = g['max_page_chars']
        # Manche Modelle (z.B. gemma4:e4b) unterstützen in Ollama kein Structured Output (HTTP 501);
        # dann wird das JSON-Format im Prompt verlangt und die Antwort selbst geparst und geprüft
        self.structured = True

    def _ask(self, prompt: str, schema: dict, image: str | None = None) -> dict:
        if self.structured:
            try:
                return json.loads(self._chat(prompt, image, schema))
            except ollama.ResponseError as e:
                if e.status_code != 501:
                    raise
                self.structured = False
        fields = ', '.join(f'"{k}" ({v["type"]})' for k, v in schema['properties'].items())
        prompt += f'\n\nAntworte ausschließlich mit einem JSON-Objekt mit den Feldern {fields}, ohne weiteren Text.'
        return _parse_json(self._chat(prompt, image, None), schema)

    def _chat(self, prompt: str, image: str | None, schema: dict | None) -> str:
        msg = {'role': 'user', 'content': prompt}
        if image:
            msg['images'] = [image]
        resp = ollama.chat(model=self.model, messages=[msg], format=schema,
                           options=self.options, think=self.think)
        return resp.message.content

    def text_question(self, text: str) -> dict:
        prompt = TEXT_PROMPT.format(text=text[:self.max_chars], rules=COMMON_RULES)
        return self._ask(prompt, TEXT_SCHEMA)

    def visual_question(self, text: str, image: str) -> dict:
        prompt = VISUAL_PROMPT.format(text=text[:self.max_chars] or '(kein Text)', rules=COMMON_RULES)
        return self._ask(prompt, VISUAL_SCHEMA, image=image)

    def answerable_from_text(self, text: str, question: str, answer: str) -> dict:
        prompt = FILTER_PROMPT.format(text=text[:self.max_chars], question=question, answer=answer)
        return self._ask(prompt, FILTER_SCHEMA)


def process(gen: Generator, page: dict, text: str) -> dict:
    rec = {'key': page_key(page['doc_id'], int(page['page'])), 'type': page['type'], 'model': gen.model}
    t0 = time.perf_counter()
    if page['type'] == 'text':
        rec['generation'] = gen.text_question(text)
        rec['filter'] = None
        rec['filter_result'] = 'not_applicable'
    else:
        rec['generation'] = gen.visual_question(text, page['image_path'])
        if not rec['generation'].get('usable', True):
            rec['filter'] = None
            rec['filter_result'] = 'no_visual_content'
        elif page['subtype'] == 'scan':
            rec['filter'] = None
            rec['filter_result'] = 'skipped_scan'
        else:
            g = rec['generation']
            rec['filter'] = gen.answerable_from_text(text, g['question'], g['answer'])
            rec['filter_result'] = ('answerable_from_text' if rec['filter']['answerable']
                                    else 'not_answerable_from_text')
    rec['seconds'] = round(time.perf_counter() - t0, 2)
    return rec


def draft_row(i: int, page: dict, rec: dict, text: str) -> dict:
    g = rec['generation']
    keep = int(rec['filter_result'] in ('not_applicable', 'skipped_scan', 'not_answerable_from_text'))
    return {
        'id': f'q{i:03d}',
        'question': g['question'].strip(),
        'answer': g['answer'].strip(),
        'doc_id': page['doc_id'],
        'page': page['page'],
        'type': page['type'],
        'page_image_path': page['image_path'],
        'filter_result': rec['filter_result'],
        'keep': keep,
        'subtype': page['subtype'],
        'gold_pages': page['gold_pages'],
        'figure': g.get('figure', ''),
        'filter_reason': (rec['filter'] or {}).get('reason', ''),
        'max_verbatim_words': max_verbatim_words(g['question'], text),
        'notes': '',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, help='Smoke-Test: nur die ersten N Seiten (eigenes Verzeichnis)')
    parser.add_argument('--overwrite-draft', action='store_true',
                        help='questions_draft.csv überschreiben (manuelle Änderungen gehen verloren!)')
    args = parser.parse_args()
    cfg = load_config()
    out = data_dir(cfg, args.limit)

    if (ROOT / cfg['paths']['data_dir'] / 'testset.json').exists() and not args.limit:
        raise SystemExit('Testset ist bereits eingefroren – Fragen werden nicht neu generiert.')
    draft_path = out / 'questions_draft.csv'
    if draft_path.exists() and not args.overwrite_draft and not args.limit:
        raise SystemExit(f'{draft_path} existiert (evtl. mit manuellen Änderungen). '
                         'Nur mit --overwrite-draft neu schreiben.')

    with open(out / 'selected_pages.csv', encoding='utf-8') as f:
        pages = list(csv.DictReader(f))
    if args.limit:
        pages = pages[:args.limit]

    cache_path = out / 'generation_cache.jsonl'
    gen = Generator(cfg)
    # nur Einträge des aktuell konfigurierten Modells verwenden, damit keine Modelle gemischt werden
    cache = {r['key']: r for r in read_jsonl(cache_path) if r.get('model') == gen.model}
    model_digest = next((m.digest for m in ollama.list().models if m.model == gen.model), 'unknown')
    write_json(out / 'generation_meta.json', run_metadata(
        model=gen.model, model_digest=model_digest, options=gen.options, think=gen.think,
        structured_output='automatisch: JSON-Schema, sonst JSON-Anweisung im Prompt + Validierung',
        ollama=__import__('common').ollama_version(), prompts={
            'text': TEXT_PROMPT, 'visual': VISUAL_PROMPT, 'filter': FILTER_PROMPT, 'rules': COMMON_RULES,
        }))

    docs_dir = ROOT / cfg['paths']['docs_dir']
    texts = {}
    for n, page in enumerate(pages, 1):
        key = page_key(page['doc_id'], int(page['page']))
        texts[key] = page_text(page['doc_id'], int(page['page']), docs_dir)
        if key in cache:
            continue
        try:
            rec = process(gen, page, texts[key])
        except Exception as e:  # einzelne Fehlschläge nicht cachen, beim nächsten Lauf neu versuchen
            print(f'[{n}/{len(pages)}] FEHLER {key}: {e}')
            continue
        cache[key] = rec
        append_jsonl(cache_path, rec)
        print(f"[{n}/{len(pages)}] {rec['seconds']:5.1f}s {page['type']:6} {rec['filter_result']:26} "
              f"{rec['generation']['question'][:90]}")

    # Modell entladen, damit Stufe 3 (Qwen3-VL auf MPS) den Speicher bekommt
    ollama.generate(model=gen.model, prompt='', keep_alive=0)

    rows = [draft_row(i, p, cache[k], texts[k]) for i, p in enumerate(pages, 1)
            if (k := page_key(p['doc_id'], int(p['page']))) in cache]
    with open(draft_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=DRAFT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f'\n{len(rows)}/{len(pages)} Fragen -> {draft_path}')
    print('filter_result:', dict(Counter(r['filter_result'] for r in rows)))
    print('keep=1 je Typ:', dict(Counter(r['type'] for r in rows if r['keep'])))
    flagged = sum(r['max_verbatim_words'] >= cfg['generation']['verbatim_ngram'] for r in rows)
    print(f"Fragen mit >= {cfg['generation']['verbatim_ngram']} wörtlich übernommenen Wörtern: {flagged}")
    if len(rows) < len(pages):
        print('Nicht alle Seiten verarbeitet – Skript erneut starten (Cache wird fortgesetzt).')


if __name__ == '__main__':
    main()
