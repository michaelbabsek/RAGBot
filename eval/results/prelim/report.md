# Retrieval-Evaluation RAGBot

> ⚠️ **VORLÄUFIG** – Testset mit ungeprüften keep-Vorschlägen, keine manuelle Durchsicht. Zahlen nicht für den Lebenslauf verwenden.

Testset: N = 224 generierte Fragen (text 120, visual 104). Treffer = Gold-Seite inkl. inhaltsgleicher Kopien; 95-%-KI per Bootstrap (10000 Resamples, Seed 42); p-Werte: exakter McNemar-Test (zweiseitig).

## 1. CV-Zahlen

| Größe | Wert | 95-%-KI | Basis |
|---|---|---|---|
| N | 224 | – | alle Fragen |
| n_visual | 104 | – | visuelle Fragen |
| X = Recall@4 MM-0 (ohne Reranking) | 76.8 % | [71.0; 82.1] | alle |
| Y = Recall@4 MM-R (mit Reranking) | 87.1 % | [82.6; 91.5] | alle |
| A = Recall@4 TXT-R (nur Text) | 47.1 % | [37.5; 56.7] | visual |
| B = Recall@4 MM-R (Text + Seitenbilder) | 75.0 % | [66.3; 82.7] | visual |

- Reranking (MM-0 → MM-R, alle Fragen): Δ = +10.3 Pp., 95-%-KI [6.2; 14.3] Pp., p = < 0.001 (signifikant bei α = 0.05); 23 Fragen verbessert, 0 verschlechtert.
- Seitenbilder (TXT-R → MM-R, visuelle Fragen): Δ = +27.9 Pp., 95-%-KI [19.2; 36.5] Pp., p = < 0.001 (signifikant bei α = 0.05); 30 Fragen verbessert, 1 verschlechtert.

Ausgefüllte Lebenslauf-Zeile:

> Evaluated retrieval on 224 generated test questions: re-ranking raised Recall@4 from 77% to 87%; page images lifted it from 47% to 75% on visual questions compared with text-only search

Zum Vergleich das Produktivverhalten mit Score-Schwelle (min_score = 0.0): Recall@4 = 86.2 % [81.2; 90.6], im Mittel 3.13 weitergegebene Items.

## 2. Alle Konfigurationen

| Konfiguration | Teilmenge | n | Recall@4 [95-%-KI] | Recall@4 strikt | Recall@16 [95-%-KI] | MRR@16 [95-%-KI] | Ø Items |
|---|---|---|---|---|---|---|---|
| MM-0 | all | 224 | 76.8 % [71.0; 82.1] | 76.8 % | 90.2 % [86.2; 93.8] | 0.682 [0.628; 0.734] | 4.00 |
| MM-0 | text | 120 | 95.0 % [90.8; 98.3] | 95.0 % | 98.3 % [95.8; 100.0] | 0.864 [0.813; 0.911] | 4.00 |
| MM-0 | visual | 104 | 55.8 % [46.2; 64.4] | 55.8 % | 80.8 % [73.1; 87.5] | 0.471 [0.392; 0.551] | 4.00 |
| MM-R | all | 224 | 87.1 % [82.6; 91.5] | 87.1 % | 90.2 % [86.2; 93.8] | 0.827 [0.780; 0.871] | 4.00 |
| MM-R | text | 120 | 97.5 % [94.2; 100.0] | 97.5 % | 98.3 % [95.8; 100.0] | 0.929 [0.890; 0.964] | 4.00 |
| MM-R | visual | 104 | 75.0 % [66.3; 82.7] | 75.0 % | 80.8 % [73.1; 87.5] | 0.708 [0.623; 0.785] | 4.00 |
| TXT-0 | all | 224 | 67.9 % [61.6; 74.1] | 67.9 % | 82.6 % [77.2; 87.5] | 0.612 [0.555; 0.669] | 4.00 |
| TXT-0 | text | 120 | 95.0 % [90.8; 98.3] | 95.0 % | 98.3 % [95.8; 100.0] | 0.866 [0.815; 0.913] | 4.00 |
| TXT-0 | visual | 104 | 36.5 % [26.9; 46.2] | 36.5 % | 64.4 % [54.8; 73.1] | 0.319 [0.244; 0.394] | 4.00 |
| TXT-R | all | 224 | 74.6 % [68.8; 80.4] | 74.1 % | 82.6 % [77.2; 87.5] | 0.693 [0.637; 0.748] | 4.00 |
| TXT-R | text | 120 | 98.3 % [95.8; 100.0] | 98.3 % | 98.3 % [95.8; 100.0] | 0.919 [0.879; 0.954] | 4.00 |
| TXT-R | visual | 104 | 47.1 % [37.5; 56.7] | 46.2 % | 64.4 % [54.8; 73.1] | 0.432 [0.348; 0.517] | 4.00 |
| MM-R (min_score=0) | all | 224 | 86.2 % [81.2; 90.6] | 86.2 % | 90.2 % [86.2; 93.8] | 0.827 [0.780; 0.871] | 3.13 |
| MM-R (min_score=0) | text | 120 | 97.5 % [94.2; 100.0] | 97.5 % | 98.3 % [95.8; 100.0] | 0.929 [0.890; 0.964] | 3.70 |
| MM-R (min_score=0) | visual | 104 | 73.1 % [64.4; 81.7] | 73.1 % | 80.8 % [73.1; 87.5] | 0.708 [0.623; 0.785] | 2.48 |

Gepaarte Vergleiche (Recall@4):

| Vergleich | Teilmenge | n | Δ [95-%-KI] | verbessert | verschlechtert | p (McNemar) |
|---|---|---|---|---|---|---|
| MM-0 → MM-R | all | 224 | +10.3 Pp. [6.2; 14.3] | 23 | 0 | < 0.001 |
| MM-0 → MM-R | text | 120 | +2.5 Pp. [0.0; 5.8] | 3 | 0 | 0.250 |
| MM-0 → MM-R | visual | 104 | +19.2 Pp. [11.5; 26.9] | 20 | 0 | < 0.001 |
| TXT-0 → TXT-R | all | 224 | +6.7 Pp. [3.1; 10.3] | 17 | 2 | < 0.001 |
| TXT-R → MM-R | visual | 104 | +27.9 Pp. [19.2; 36.5] | 30 | 1 | < 0.001 |
| TXT-R → MM-R | text | 120 | -0.8 Pp. [-2.5; 0.0] | 0 | 1 | 1.000 |
| TXT-0 → MM-0 | visual | 104 | +19.2 Pp. [11.5; 26.9] | 21 | 1 | < 0.001 |

Hinweis: 5 visuelle Fragen stammen von Scans ohne Text-Chunk; für TXT-* sind sie per Konstruktion nicht auffindbar.

Latenz pro Anfrage (ms):

| Konfiguration | Retrieval p50 | Retrieval p95 | Reranking p50 | Reranking p95 | Gesamt p50 | Gesamt p95 |
|---|---|---|---|---|---|---|
| MM-0 | 91 | 118 | 0 | 0 | 91 | 118 |
| MM-R | 129 | 313 | 19213 | 35029 | 19340 | 35163 |
| TXT-0 | 101 | 130 | 0 | 0 | 101 | 130 |
| TXT-R | 122 | 152 | 6785 | 13727 | 6907 | 13880 |

## 3. Beispiele

### Reranking hat geholfen (MM-0 → MM-R)

**q170** (visual/figure) – Welche spezifischen Thread-IDs sind in der Liste der Thread-Status-Anzeige für Thread-0 und Thread-1 zu sehen?  
Gold-Seite: VS-03-Threads.pdf S. 14

- vorher (MM-0): · VS-03-Threads.pdf S. 9 (Text); · VS-03-Threads.pdf S. 16 (Text); · VS-03-Threads.pdf S. 23 (Text); · VS-03-Threads.pdf S. 10 (Text)
- nachher (MM-R): · VS-03-Threads.pdf S. 9 (Bild); · VS-03-Threads.pdf S. 24 (Text); ✅ VS-03-Threads.pdf S. 14 (Text); · VS-03-Threads.pdf S. 7 (Bild)

**q118** (visual/figure) – Was sagt der Qualitätsmanager in der Sprechblase über seine Rolle im Projekt?  
Gold-Seite: SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 11

- vorher (MM-0): · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 2 (Text); · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 10 (Text); · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 17 (Text); · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 9 (Text)
- nachher (MM-R): ✅ SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 11 (Bild); · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 2 (Bild); · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 2 (Text); · SEI - 06.1 - Qualitätsmanagement. Grundlagen.pdf S. 9 (Text)

**q167** (visual/figure) – Welche spezifischen Epics und User Stories sind im gezeigten Jira Board sichtbar?  
Gold-Seite: SEI - 04.2 - Projektmanagement. Agiles Managment.pdf S. 14

- vorher (MM-0): · SEI - 04 - Übung 4 (Projektmanagement + Java) (1).pdf S. 1 (Text); · SEI - 04.3 - Projektmanagement. Planning Poker 2026.pdf S. 3 (Text); · SEI - 04.4 - Projektmanagement. Jira 2026.pdf S. 3 (Text); · SEI - 04.4 - Projektmanagement. Jira 2026.pdf S. 4 (Text)
- nachher (MM-R): · SEI - 04.4 - Projektmanagement. Jira 2026.pdf S. 32 (Bild); ✅ SEI - 04.2 - Projektmanagement. Agiles Managment.pdf S. 14 (Bild); · SEI - 03.2 - Requirements Engineering. Methoden.pdf S. 23 (Text); · SEI - 04.3 - Projektmanagement. Planning Poker 2026.pdf S. 3 (Bild)

### Seitenbilder haben geholfen (TXT-R → MM-R)

**q052** (visual/figure) – Welche spezifische Methode wird im UDPClient-Code verwendet, um die Zeit zu formatieren, und welche Formatzeichenfolge wird dabei genutzt?  
Gold-Seite: VS-06-Sockets.pdf S. 15

- vorher (TXT-R): · VS-02-Netzwerkkommunikation.pdf S. 20 (Text); · VS-06-Sockets.pdf S. 22 (Text); · VS-06-Sockets.pdf S. 7 (Text); · VS-06-Sockets.pdf S. 21 (Text)
- nachher (MM-R): ✅ VS-06-Sockets.pdf S. 15 (Bild); · VS-06-Sockets.pdf S. 22 (Bild); · VS-06-Sockets.pdf S. 20 (Bild); · VS-02-Netzwerkkommunikation.pdf S. 20 (Text)

**q180** (visual/figure) – Welche Methode wird in der Klasse SocketClient verwendet, um die Verbindung zu einem Client zu senden?  
Gold-Seite: VS-06-Sockets.pdf S. 7

- vorher (TXT-R): · VS-06-Sockets.pdf S. 18 (Text); · VS-06-Sockets.pdf S. 10 (Text); · VS-06-Sockets.pdf S. 6 (Text); · VS-06-Sockets.pdf S. 13 (Text)
- nachher (MM-R): ✅ VS-06-Sockets.pdf S. 7 (Bild); · VS-06-Sockets.pdf S. 18 (Text); · VS-06-Sockets.pdf S. 10 (Text); · VS-06-Sockets.pdf S. 6 (Text)

**q062** (visual/figure) – Welche spezifische Java-Version erfordert, dass der SecurityManager entfernt wurde, da die Stub-Klassen lokal vorhanden sein müssen?  
Gold-Seite: VS-04-RMI.pdf S. 19

- vorher (TXT-R): · VS-04-RMI.pdf S. 35 (Text); · VS-04-RMI.pdf S. 18 (Text); · VS-04-RMI.pdf S. 22 (Text); · VS-04-RMI.pdf S. 34 (Text)
- nachher (MM-R): · VS-04-RMI.pdf S. 35 (Text); · VS-04-RMI.pdf S. 35 (Bild); ✅ VS-04-RMI.pdf S. 19 (Bild); · VS-04-RMI.pdf S. 18 (Bild)

### Seitenbilder haben geschadet (TXT-R → MM-R)

**q025** (visual/diagram) – In welcher Phase des Projektverlaufs wird die Aktivität 'codieren' durchgeführt?  
Gold-Seite: SEM - 02 - Lösung (Vorgehensmodelle).pdf S. 4

- vorher (TXT-R): · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 15 (Text); · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 14 (Text); · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 9 (Text); ✅ SEM - 02 - Lösung (Vorgehensmodelle).pdf S. 4 (Text)
- nachher (MM-R): · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 14 (Bild); · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 15 (Bild); · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 15 (Text); · SEI - 02.1 - Vorgehensmodelle. Grundlagen.pdf S. 14 (Text)

