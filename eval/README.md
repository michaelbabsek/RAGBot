# Retrieval-Evaluation

Misst, wie stark Reranking und Seitenbilder das Retrieval von RAGBot verbessern (Recall@4, Recall@16, MRR@16).
Relevanzeinheit ist die Seite (`doc_id`, `page`), inhaltsgleiche Kopien zählen mit (Äquivalenzklasse).
Am Code in `ragbot/` wird nichts verändert. Die Skripte nutzen dieselben Komponenten und laufen gegen eine eingefrorene Kopie des Index.

| Konfiguration | Index | Reranking |
|---|---|---|
| MM-0 | Text + Seitenbilder | nein (Top-4 nach Embedding) |
| MM-R | Text + Seitenbilder | ja (aktuelles System, ohne Score-Schwelle; mit Schwelle als Zusatzzeile) |
| TXT-0 | nur Text-Chunks | nein |
| TXT-R | nur Text-Chunks | ja |

Alle Befehle werden im Repo-Root ausgeführt (`pip install -r eval/requirements.txt`, Ollama mit `gemma4:e4b`).
Jede Stufe zuerst mit `--limit 5` als Smoke-Test laufen lassen. Die Ausgaben landen dann in `eval/*/smoke/`.

## Reihenfolge

```bash
# 0. Index einfrieren (einmalig) – danach nie neu bauen; prüfen mit --verify
python eval/freeze_index.py

# 1. Seiten analysieren und stratifiziert ziehen  -> eval/data/pages.csv, selected_pages.csv
python eval/select_pages.py

# 2. Fragen generieren (gemma4:e4b, resumable)  -> eval/data/questions_draft.csv
python eval/generate_questions.py
```

**⏸ Checkpoint: Fragen manuell prüfen.** In `questions_draft.csv` pro Zeile `keep` auf 0 oder 1 setzen. `question` und `answer` dürfen umformuliert werden.
Worauf achten:
- Ist die Frage ohne die Seite verständlich?
- Ist sie eindeutig dieser Seite zuzuordnen?
- Bei `visual`: Ist sie wirklich nur über die Abbildung beantwortbar?
- Spalte `max_verbatim_words` ≥ 5: Hat das Modell wörtlich abgeschrieben?

Fehlklassifizierte Seiten können optional in `eval/data/page_overrides.csv` (`doc_id,page,page_class`) korrigiert werden.

```bash
# Testset einfrieren (nur keep=1). Danach ist es fix und wird nicht neu generiert.
python eval/freeze_testset.py

# 3. Retrieval-Läufe (resumable)  -> eval/runs/<config>.jsonl
python eval/run_retrieval.py --config all     # oder einzeln: --config MM-R

# 4. Auswertung  -> eval/results/report.md, per_question.csv, metrics.json
python eval/report.py
```

## Vorläufiger Lauf (`--prelim`)
`freeze_testset.py`, `run_retrieval.py` und `report.py` akzeptieren `--prelim`. Dann wird das Testset direkt aus den keep-Vorschlägen gebaut, ohne manuelle Durchsicht, und alles landet in `prelim/`-Unterverzeichnissen. Das echte Einfrieren wird dadurch nicht blockiert. Die im Projekt-README berichteten Ergebnisse stammen aus diesem Lauf:

```bash
python eval/freeze_testset.py --prelim
python eval/run_retrieval.py --config all --prelim
python eval/report.py --prelim
```

## Hinweise
- Die Stufen 2 und 3 nie gleichzeitig laufen lassen, weil Gemma und Qwen3-VL sich den RAM teilen. Stufe 2 entlädt das Gemma-Modell am Ende selbst. gemma4:26b (18 GB) hat 24 GB RAM überlastet, deshalb wird gemma4:e4b verwendet.
- Alle LLM-Aufrufe laufen mit `temperature=0` und festem Seed. Embedding und Reranking laufen in bf16 auf MPS, dort sind minimale numerische Abweichungen zwischen Läufen möglich.
- Ein Smoke-Lauf von `run_retrieval.py` prüft zusätzlich, dass MM-R mit Score-Schwelle exakt dasselbe liefert wie `ragbot.Retriever`.
