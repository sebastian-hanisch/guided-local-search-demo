# Guided Local Search – eine Lieferrunde, die unliebsame Kanten teurer macht – Streamlit-Demo

**[→ Demo live ausprobieren](#) (Deploy offen)**

Zwölftes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
dieselbe Rundtour wie in der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) und der [tabu-search-demo](https://sebastianhanisch-tabu-search-demo.streamlit.app/) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke, dieselbe volle-2-opt-Nachbarschaft und dieselbe Bewertungs-Zählweise wie Tabu Search.

**Einordnung in die Reihe:** **Guided Local Search** (Voudouris & Tsang 1999) ist ein direktes Kind von Hill Climbing, wie Simulated Annealing, ILS/VNS, Tabu Search und GRASP - und der direkte **Kontrast** zu Tabu Search: dasselbe Problem (über ein lokales Optimum hinauskommen, ohne Struktur wegzuwerfen), der entgegengesetzte Mechanismus. Tabu Search verbietet die zuletzt entfernten Kanten für eine feste Zahl Iterationen (Gedächtnis, das **verfällt**). Guided Local Search bestraft stattdessen Tourkanten in der Zielfunktion selbst (`Länge + λ · Strafen`) - eine Strafe, die **nie verfällt**. Googles OR-Tools-Routing-Solver nutzt genau dieses Verfahren als Standard-Metaheuristik für Fahrzeugrouting.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [gebaut]
  ├─ tabu-search-demo (immer der beste Zug, verfallendes Gedächtnis)             [gebaut]
  ├─ grasp-demo (randomisierte Konstruktion, viele Starts)                       [gebaut]
  └─ guided-local-search-demo (immer der beste Zug, nie verfallende Strafe)      [dieses Stück]
```

Ergebnis in Kürze: **Unter 100 Tausend Vorschlägen ist Guided Local Search mit Tabu Search nahezu IDENTISCH - ab 200 Tausend zieht es klar davon, und der Vorsprung wächst mit jeder Budget-Verdopplung, statt zu schrumpfen.** 60 Stopps, Lambda 5 (kalibriert): bei 10-50 Tausend Vorschlägen liegen beide Verfahren auf zwei Nachkommastellen genau gleichauf (**287.75 %** gegen Tabus **287.8 %** bei 10 Tausend) - beide sind volle-Nachbarschaft-Verfahren, und bei so wenigen Iterationen hat die dauerhafte Strafe noch keine Gelegenheit, sich von Tabus verfallender Sperre abzusetzen. Bei 100 Tausend liegt Tabu Search noch knapp vorn (7.9 % gegen 8.53 %). Ab **200 Tausend** dreht sich das Bild und bleibt gedreht: **3.39 %** gegen Tabus 4.19 %, bei 500 Tausend **1.56 %** gegen 3.1 %, bei 2 Millionen **0.52 %** gegen 1.9 % - ein wachsender, nicht schrumpfender Vorsprung, anders als Tabu Searchs eigener (schrumpfender) Vorsprung vor Hill Climbing mit Neustarts.
**Lambda** (Gewicht der Strafe) hat denselben Sweet Spot wie Tabus Tenure - nicht "mehr Strafe ist sicherer": bei Lambda 0.1 (zu schwach) verhält sich die Suche fast wie ein einzelner Hill-Climbing-Abstieg (7.74 % gegen 7.85 %), bei Lambda 30 (zu stark) dominiert Vermeidung über Kürze (5.69 %) - das Minimum liegt bei **Lambda 5** (3.39 %).
Bei größeren Instanzen (ab ~100-150 Stopps) verschwindet der Vorteil wieder: das für 60 Stopps kalibrierte Budget reicht für KEINES der beiden Verfahren mehr, um über die ersten Züge hinauszukommen - dort sind Guided Local Search und Tabu Search wieder auf zwei Nachkommastellen genau identisch (derselbe geteilte Budget-Engpass, kein Unterschied im Mechanismus).

| Frage | Ergebnis (60 gleichverteilte Stopps, Lambda 5, zufällige Startlösung; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ✅ Guided Local Search **3.39 %** über der Schranke gegen **4.19 %** für Tabu Search (gleiches Budget), **7.85 %** für einen Hill-Climbing-Abstieg, **4.88 %** für Hill Climbing mit Neustarts |
| **Budget, klein** | ➖ Bei 10 / 25 / 50 Tausend Vorschlägen: **287.75 / 155.09 / 54.73 %** - praktisch IDENTISCH mit Tabu Search (287.8 / 155.1 / 54.7 %); die Strafe hat noch keine Gelegenheit zu wirken |
| **Budget, groß** | ✅❗ Bei 200 Tausend / 500 Tausend / 1 / 2 Millionen: **3.39 / 1.56 / 0.86 / 0.52 %** gegen Tabus 4.2 / 3.1 / 2.3 / 1.9 % - ein Vorsprung, der mit jeder Budget-Verdopplung WÄCHST, statt zu schrumpfen |
| **Lambda** | ⚠️ 0.1 / 1.0 / 2.0 / 5.0 / 15.0 / 30.0: **7.74 / 5.83 / 4.65 / 3.39 / 4.28 / 5.69 %** - ein Sweet Spot bei 5, kein "mehr Strafe ist sicherer" |
| **Größe** | ➖ 200 Stopps, 1 Million Vorschläge (5 Tausend · Stopps): **306.74 %**, praktisch identisch mit Tabu Search bei gleichem Budget (306.74 %) - derselbe geteilte Budget-Engpass, KEIN Unterschied im Mechanismus |
| **Startlösung** | ✅ Nächster Nachbar **1.09 %** gegen zufällig **3.39 %** - wie bei Tabu Search zählt die Startlösung hier deutlich |

## Was die Demo zeigt

1. **Guided Local Search in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Suche** (Iterations-Regler + ▶️ Suche abspielen: Länge der aktuellen/besten Tour über die bewerteten Nachbarn, mit Tabu Search als Referenzlinie, dazu die Tour nach der gewählten Iteration) → **Ergebnis** (beste Tour neben der besten aus Tabu Search).
2. **Was die Suche gefunden hat:** beste und letzte Tour, Tabu Search bei gleichem Budget, Hill Climbing mit Neustarts, Anteil Strafen statt Zug; Urteil (`beats_tabu` → `comparable` → `tabu_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, Lambda, Stopps, Gruppen und Startlösung (feste Instanzen ab 100000, drei Ketten je Instanz), jeweils gegen Tabu Search als Referenzlinie.
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen (wann zieht die dauerhafte Strafe an Tabu Search vorbei?); Streuung über 20 Ketten; Skalierung von 20 bis 200 Stopps.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (das Budget reicht für genug teure Iterationen, Lambda passt zur Instanz, jede Iteration ist bezahlbar, die Strafe braucht mehrere lokale Optima).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Lambda** (0.05–30), **Budget** (10 Tausend bis 2 Millionen bewertete Nachbarn), **Startlösung**, Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).
Kein Nachbarschafts-Regler (bewusst nur 2-opt, wortgleich zu Tabu Search - damit der Budgetvergleich fair ist); kein Zufalls-Seed für die Suche selbst (Guided Local Search ist deterministisch, nur die Startlösung streut).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_tabu / comparable / tabu_wins |
| Zu kleines Budget (25 Tausend) | comparable |
| Zu kleine Strafe (Lambda 0.1) | beats_tabu / comparable / tabu_wins |
| Zu große Strafe (Lambda 30) | beats_tabu / comparable / tabu_wins |
| Nächster Nachbar als Start | beats_tabu / comparable / tabu_wins |
| Großes Budget (1 Million) | beats_tabu / comparable |
| Große Instanz (200 Stopps, 1 Million) | comparable |

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`gls_scenario.py`, `gls_tour.py`): wortgleiche Kopien aus der [tabu-search-demo](https://sebastianhanisch-tabu-search-demo.streamlit.app/) (per Test gegen eingefrorene Werte) - dieselbe volle, vektorisierte 2-opt-Bewertung, damit der Budgetvergleich mit Tabu Search fair ist (dieselbe Kosten-Buchführung, keine Kandidatenliste).
- **Guided-Local-Search-Schleife** (`gls_algorithm.py`): jede Iteration wird die volle 2-opt-Nachbarschaft auf der ERWEITERTEN Zielfunktion `Länge + λ · Strafen` bewertet (vektorisiert); verbessert kein Zug die erweiterte Zielfunktion (erweitertes lokales Optimum), wird die Tourkante mit dem höchsten Nutzen `Länge / (1 + Strafe)` bestraft (Strafzahl + 1). Strafen verfallen nie. Die beste Tour wird immer anhand der echten, unveränderten Länge verfolgt. Deterministisch - kein Zufall im Kern.
- **Tabu Search als Vergleichspartner** (`gls_tabu_algorithm.py`): byte-identische Kopie von `tabu_algorithm.py` aus der tabu-search-demo (kalibrierte Tenure 20) - läuft bei jeder Analyse live mit, aus derselben Startlösung, demselben Budget, derselben Instanz.
- **Hill Climbing mit Neustarts** (`gls_evaluation.py`): Abstiege aus zufälligen Startlösungen, voller Rescan (wie in der Wurzel-Demo), bis das Budget erreicht ist.
- **Auswertung** (`gls_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "eine dauerhafte Strafe schlägt eine verfallende Sperre bei gleichem Budget"** – **weder pauschal bestätigt noch widerlegt, sondern budgetabhängig UND größenabhängig**: bei knappem Budget (unter 100 Tausend, 60 Stopps) sind beide Verfahren praktisch IDENTISCH - die Strafe braucht mehrere lokale Optima, um überhaupt zu wirken, und bei nur 6-29 Iterationen kommt keines der beiden Verfahren dorthin. Erst ab 200 Tausend zeigt sich der Unterschied, und dort wächst der Vorsprung mit dem Budget, statt zu schrumpfen (anders als Tabu Searchs eigener Vorsprung vor Hill Climbing mit Neustarts, der sich bei 2 Millionen wieder auf null annähert). Bei größeren Instanzen (ab ~100-150 Stopps, dasselbe für 60 Stopps kalibrierte Budget) verschwindet der Unterschied wieder komplett - beide Verfahren sind dann gleichermaßen unterversorgt.
- **Lambda ist ein Sweet-Spot-Parameter**, nicht "mehr Strafe ist sicherer" - Lambda 5 schlägt sowohl Lambda 0.1 (zu schwach, praktisch wie ein einzelner Abstieg) als auch Lambda 30 (zu stark, Vermeidung dominiert über Kürze). Dieselbe Lehre wie SA's Temperatur, ILS' Störstärke, VNS' k_max, Tabus Tenure und GRASPs RCL-Größe, hier zum siebten Mal bestätigt.
- **Eine gute Startlösung hilft deutlich** (Nächster Nachbar 1.09 % gegen zufällig 3.39 %) - wie bei Tabu Search, weil bei nur rund 110-570 teuren Iterationen jeder Vorsprung zählt.
- **Skaliert NICHT besser als Tabu Search** mit der Instanzgröße - bei 200 Stopps kostet eine Iteration rund 20 Tausend Bewertungen wie bei Tabu Search, derselbe Preis für die volle Nachbarschaft. Eine Kandidatenlisten-Variante wäre der naheliegende nächste Schritt, hier bewusst nicht gebaut (bewusste Scope-Entscheidung, damit der Vergleich mit Tabu Search sauber bleibt).
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Strafmatrix-Verwaltung:** unabhängige Nachrechnung (aus Momentaufnahmen + `debug_trace`) baut die Strafmatrix NUR anhand der öffentlichen Regel neu auf (nicht anhand der internen Datenstruktur des Algorithmus) und bestätigt: der gewählte Zug ist bei jeder Iteration die beste Verbesserung der erweiterten Zielfunktion unter den so rekonstruierten Strafen; jede Bestrafung trifft nachweislich die Kante mit dem höchsten Nutzen `Länge / (1 + Strafe)` unter den aktuellen Tourkanten.
- **"Lambda=0"-Grenzfall:** die Suche friert nachweislich nach dem ersten lokalen Optimum komplett ein (kein Pendeln wie bei Tabus Tenure 0 - ohne Gewicht in der Zielfunktion ändert eine wachsende Strafe nichts an der Zugauswahl).
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke gegen Brute-Force (n = 8) und CP-SAT (n = 20); Instanz gegen eingefrorene Werte (identisch mit tabu-search-demo).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, Lambda- und Größen-Aussagen, inklusive der "nahezu identisch unten / wachsender Vorsprung oben"-Aussage gegen Tabu Search; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung); alle 7 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Iterations-Regler, ▶️ Abspielen und ▶️ Suche abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `gls_algorithm.py` | Die Guided-Local-Search-Schleife: erweiterte Zielfunktion, Nutzen-Bestrafung |
| `gls_tabu_algorithm.py` | Tabu Search als Vergleichspartner (byte-identisch zur tabu-search-demo) |
| `gls_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `gls_scenario.py`, `gls_constants.py` | Instanzen; Konstanten, Presets |
| `gls_evaluation.py` | Analyse, Urteil, Tabu-Search-Vergleich, Hill Climbing mit Neustarts, Sweeps, Streuung, Skalierung |
| `gls_presets.py`, `gls_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Übernommener Kern, Guided-Local-Search-Schleife (unabhängige Replay-Verifikation), Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
