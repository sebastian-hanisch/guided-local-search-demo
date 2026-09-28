"""Konstanten der Guided-Local-Search-Demo: Szenario (wortgleich zur hill-climbing-demo), Regler, Beschriftungen (Presets/Messwerte folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_START = "random"
START_LABELS = {"random": "Zufällig", "nearest": "Nächster Nachbar"}
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- Guided-Local-Search-eigene Regler --------------------------------------------------------------------------------
LAMBDA_MIN, LAMBDA_MAX, DEFAULT_LAMBDA = 0.05, 30.0, 5.0
DEFAULT_BUDGET = 200000

# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), 60 gleichverteilte Stopps, Abstand zur Schranke (2026-09-28):
# Lambda-Sweep (Budget 200T): 0.05 -> 7.75 %, 0.1 -> 7.74 %, 0.2 -> 7.16 %, 0.3 -> 7.09 %, 0.5 -> 6.21 %, 1.0 -> 5.83 %, 2.0 -> 4.65 %, 3.0 -> 4.67 %,
#   4.0 -> 4.32 %, 4.5 -> 3.77 %, 5.0 -> 3.39 % (BESTES gemessen), 5.5 -> 4.17 %, 6.0 -> 4.41 %, 6.5 -> 3.77 %, 7.0 -> 3.70 %, 8.0 -> 3.86 %,
#   15.0 -> 4.28 %, 30.0 -> 5.69 %, 60.0 -> 6.10 %. Ein flaches Tal mit Minimum bei 5, keine scharfe Kante - dieselbe "nicht-monotones Optimum"-Lehre
#   wie SA's Temperatur, ILS' Störstärke, VNS' k_max, Tabus Tenure, GRASPs RCL-Größe. Bei sehr kleinem Lambda (0.1) verhält sich GLS praktisch wie ein
#   einzelner Hill-Climbing-Abstieg (7.74 % gegen 7.85 %) - die Strafe ist zu schwach, um die Suche aus dem ersten lokalen Optimum zu drängen.
# Budget-Sweep (Lambda=5.0, VOLLER Sweep 10T..2M): 10T 287.75 %, 25T 155.09 %, 50T 54.73 %, 100T 8.53 %, 200T 3.39 %, 500T 1.56 %, 1M 0.86 %, 2M 0.52 %.
#   Bei 10-50 Tausend Vorschlägen ist GLS mit Tabu Search (287.8/155.1/54.7 %) nahezu IDENTISCH - beide sind volle-Nachbarschaft-Verfahren und bei so
#   wenigen Iterationen hat GLS' Strafe noch keine Gelegenheit, sich vom ersten lokalen Optimum abzusetzen; der Engpass ist Iterationen, nicht Mechanik.
#   Ab 100 Tausend trennen sich die Kurven: bei 100T liegt Tabu Search noch knapp vorn (7.9 gegen 8.53 %), ab 200 Tausend zieht GLS klar davon und der
#   Vorsprung wächst mit jeder Verdopplung (200T: 3.39 gegen 4.2 %, 500T: 1.56 gegen 3.1 %, 1M: 0.86 gegen 2.3 %, 2M: 0.52 gegen 1.9 %) - die dauerhafte
#   Strafe (nie vergessen, anders als Tabus ablaufende Sperre) verwertet zusätzliches Budget offenbar wirksamer.
# Skalierung (fixes Budget 200T, wie beim Standardfall kalibriert): 20 Stopps 0.05 %, 40 -> 1.03 %, 60 -> 3.39 %, 100 -> 111.71 %, 150 -> 471.35 %,
#   200 -> 704.54 % - AB 100 STOPPS IDENTISCH MIT TABU SEARCH (gleiche Zahlen auf die zweite Nachkommastelle). Das für n=60 kalibrierte Budget reicht bei
#   größeren Instanzen für keine der beiden Suchen aus, um über die ersten Züge hinauszukommen - der Engpass liegt am Budget, nicht am Mechanismus.
# Skalierung (Budget 5 Tausend * Stopps, wächst mit): 20 -> 0.05 %, 40 -> 1.03 %, 60 -> 2.60 %, 100 -> 9.42 %, 150 -> 109.67 %, 200 -> 306.74 % - bei 40
#   und 60 Stopps schlägt GLS Tabu Search klar (Tabu: 2.09/3.76 %), ab 150 Stopps wieder nahezu identisch (Tabu: 109.67/306.74 %) - derselbe geteilte
#   Budget-Engpass wie oben, diesmal bei größeren, aber immer noch unterversorgten Instanzen.
# Nächster Nachbar als Start (Lambda=5.0, Budget 200T): 1.09 % gegen 3.39 % bei zufälliger Startlösung - wie bei Tabu Search (0.64 gegen 3.74 %) hilft
#   eine gute Startlösung sichtbar, weil bei rund 110-570 teuren Voll-Nachbarschafts-Iterationen jeder Startvorteil zählt.


def _preset(lam=DEFAULT_LAMBDA, budget=DEFAULT_BUDGET, n=DEFAULT_N, start=DEFAULT_START):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "lam": lam, "budget": budget, "start": start, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Zu kleines Budget (25 Tausend)": _preset(budget=25000),
    "Zu kleine Strafe (Lambda 0.1)": _preset(lam=0.1),
    "Zu große Strafe (Lambda 30)": _preset(lam=30.0),
    "Nächster Nachbar als Start": _preset(start="nearest"),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; Tabu Search und Hill Climbing bei gleichem Bewertungsbudget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Lambda 5, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 3.39 % über der Schranke - Tabu Search bei gleichem Budget 4.19 %, ein Hill-Climbing-Abstieg 7.85 %, Hill Climbing mit Neustarts 4.88 %.",
    "Zu kleines Budget (25 Tausend)": "Nur 25 Tausend Vorschläge: 155.09 % über der Schranke - praktisch identisch mit Tabu Search (155.1 %) bei gleichem Budget. Bei so wenigen der teuren Voll-Nachbarschafts-Iterationen hat die Strafe noch keine Gelegenheit zu wirken.",
    "Zu kleine Strafe (Lambda 0.1)": "Eine zu schwache Strafe verhält sich fast wie ein einzelner Hill-Climbing-Abstieg: 7.74 % über der Schranke gegen 7.85 % - ohne wirksamen Druck bleibt die Suche im ersten lokalen Optimum hängen.",
    "Zu große Strafe (Lambda 30)": "Eine zu starke Strafe verzerrt die Suche zu sehr auf Vermeidung statt Kürze: 5.69 % über der Schranke, schlechter als die kalibrierten Lambda 5 (3.39 %), aber nicht katastrophal.",
    "Nächster Nachbar als Start": "Eine gute Startlösung hilft auch Guided Local Search sichtbar: 1.09 % über der Schranke gegen 3.39 % bei zufälliger Startlösung.",
    "Großes Budget (1 Million)": "1 Million Vorschläge: 0.86 % über der Schranke - deutlich besser als Tabu Search bei gleichem Budget (2.31 %), der Vorsprung der dauerhaften Strafe wächst mit dem Budget.",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps: jede Iteration kostet rund 20 Tausend Bewertungen (n²/2), 1 Million Vorschläge reichen nur für rund 50 Iterationen - die Suche kommt praktisch nicht voran (306.74 % über der Schranke), exakt wie Tabu Search bei gleichem Budget (306.74 %).",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x mehrere Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_tabu", "comparable", "tabu_wins"},
    "Zu kleines Budget (25 Tausend)": {"comparable"},
    "Zu kleine Strafe (Lambda 0.1)": {"tabu_wins", "comparable", "beats_tabu"},
    "Zu große Strafe (Lambda 30)": {"beats_tabu", "comparable", "tabu_wins"},
    "Nächster Nachbar als Start": {"beats_tabu", "comparable", "tabu_wins"},
    "Großes Budget (1 Million)": {"beats_tabu", "comparable"},
    "Große Instanz (200 Stopps, 1 Million)": {"comparable"},
}
