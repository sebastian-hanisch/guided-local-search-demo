"""Guided Local Search - eine Lieferrunde, die unliebsame Kanten teurer macht, statt sie zu verbieten - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölftes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: direktes Kind von Hill Climbing, wie Simulated
Annealing, Iterated Local Search, Tabu Search und GRASP. Kontrast zu Tabu Search: dasselbe Problem (über ein lokales Optimum
hinauskommen, ohne Struktur wegzuwerfen), der entgegengesetzte Mechanismus - kein Gedächtnis, das nach einer festen Zahl
Iterationen verfällt, sondern eine Strafe in der Zielfunktion selbst, die NIE verfällt. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import gls_constants as C
import gls_tour as T
from gls_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, scaling_table, sweep, verdict
from gls_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from gls_visualization import build_budget, build_instance, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Guided Local Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🧭 Guided Local Search – eine Lieferrunde, die unliebsame Kanten teurer macht")
st.markdown(
    """
Wie **Tabu Search** bewertet Guided Local Search jede Iteration die **volle** Nachbarschaft und nimmt immer den besten Zug – aber
statt Kanten für eine feste Zahl Iterationen zu **verbieten** (Gedächtnis, das verfällt), bekommt jede Tourkante eine **Strafzahl**,
die nie verfällt: die Suche läuft auf einer **erweiterten** Zielfunktion `Länge + λ · Strafen`. Findet sie dort keinen
verbessernden Zug mehr, wird die Kante mit dem höchsten Nutzen (Länge geteilt durch 1 + Strafe) unter den aktuellen Tourkanten
bestraft – die Zielfunktion ändert sich, die Suche findet einen neuen Zug. Die **beste** Tour wird immer anhand der echten,
unveränderten Länge verfolgt. Googles OR-Tools-Routing-Solver bietet dieses Verfahren als wählbare Metaheuristik an und beschreibt es in der Dokumentation als meist die wirksamste für Tourenplanung.
"""
)
st.caption(
    "Zwölftes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe: dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) und der "
    "[tabu-search-demo](https://sebastianhanisch-tabu-search-demo.streamlit.app/) - ein Depot in der Mitte, n Kundenstopps in "
    "einem 100 × 100-km-Gebiet, euklidische Entfernungen; dieselbe volle-2-opt-Nachbarschaft und dieselbe Bewertungs-Zählweise wie "
    "Tabu Search, damit der Vergleich bei gleichem Budget fair ist. Simulated Annealing, ILS/VNS und GRASP sind andere Antworten "
    "auf dieselbe Schwäche der Wurzel."
)

with st.expander("So funktioniert Guided Local Search", expanded=True):
    st.markdown(
        r"""
1. **Volle Nachbarschaft.** Jede Iteration wird die komplette 2-opt-Nachbarschaft bewertet (vektorisiert) - rund n²/2 bewertete Nachbarn auf einen Schlag, wie bei Tabu Search.
2. **Erweiterte Zielfunktion.** Gesucht wird der Zug, der `Länge + λ · Strafen` am stärksten verringert - nicht die echte Länge allein. Eine Kante mit hoher Strafe erscheint der Suche "künstlich teurer", auch wenn sie kurz ist.
3. **Bestrafen statt verbieten.** Findet die erweiterte Suche keinen verbessernden Zug mehr (ein erweitertes lokales Optimum), wird die Kante mit dem höchsten Nutzen `Länge / (1 + Strafe)` unter den aktuellen Tourkanten bestraft (Strafzahl + 1) - lange, noch nie bestrafte Kanten zuerst.
4. **Kein Verfall.** Anders als Tabu Searchs Sperrfrist verfällt eine Strafe NIE - eine Kante, die immer wieder im Weg steht, wird immer teurer, egal wie lange das her ist.
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke**, wie in den Geschwister-Demos. Verglichen wird vor allem mit **Tabu Search** bei gleichem Bewertungsbudget, außerdem mit einem Hill-Climbing-Abstieg und Hill Climbing mit Neustarts.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Jede Iteration kostet rund n²/2 Bewertungen - bei 200 Stopps schon rund 20 Tausend je Iteration.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    lam = st.slider(
        "Lambda (Strafgewicht)", C.LAMBDA_MIN, C.LAMBDA_MAX, key="lambda_slider",
        help="Gewicht der Strafen in der erweiterten Zielfunktion. Bei 200 Tausend Vorschlägen: 0.1 → 7.74 %, 1.0 → 5.83 %, "
             "2.0 → 4.65 %, **5.0 → 3.39 %** (Voreinstellung, bestes gemessen), 8.0 → 3.86 %, 15.0 → 4.28 %, 30.0 → 5.69 % über der "
             "Schranke - ein Sweet Spot, nicht 'mehr Strafe ist sicherer': bei zu starker Strafe dominiert Vermeidung statt Kürze.",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarn)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen liegt die beste Tour **287.75 / 155.09 / 54.73 / 8.53 / 3.39 / "
             "1.56 / 0.86 / 0.52 %** über der Schranke. Unter 100 Tausend praktisch identisch mit Tabu Search - erst ab 200 Tausend zieht "
             "die dauerhafte Strafe klar davon, und der Abstand zur Schranke schrumpft mit jeder Budget-Verdopplung stärker als bei Tabu Search.",
    )
    start = st.radio(
        "Startlösung", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k], horizontal=True,
        help="Zufällige Reihenfolge oder Nächster Nachbar. Wie bei Tabu Search zählt eine gute Startlösung sichtbar: bei nur rund "
             "110 teuren Iterationen (200 Tausend Vorschläge) ist ein Vorsprung schwerer aufzuholen (1.09 % gegen 3.39 %).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert nur die zufällige Startlösung - Guided Local Search selbst ist deterministisch (immer der beste Zug unter der erweiterten Zielfunktion, kein Zufall).",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "lambda_slider": float(lam),
    "budget_select": int(budget), "start_radio": start, "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), float(lam), int(budget), start, int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings

# --- Guided Local Search in Aktion ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Guided Local Search in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Suche", 3: "3 · Ergebnis"}
if "gls_step" not in st.session_state or st.session_state.get("gls_step_owner") != data_key:
    st.session_state["gls_step"] = 1
    st.session_state["gls_step_owner"] = data_key
    st.session_state.pop("gls_iter", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="gls_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.snapshots)
iteration = n_snaps - 1
play_search = False
if step == 2 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        iteration = st.slider("Iteration", 0, n_snaps - 1, value=n_snaps - 1, key="gls_iter", help="Die Tour nach dieser Iteration (0 = Startlösung, vor der ersten Iteration).")
    with itplay_col:
        play_search = st.button("▶️ Suche abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            st.markdown("**Länge der aktuellen und der besten Tour über die bewerteten Nachbarn**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_length, run.trace_best, a.bound, a.hc.length, T.tour_length(a.hcr_tour, a.D), tabu_length=a.tabu.best_length), width="stretch", key=f"s2_trace_{it}")
            st.markdown(f"**Tour nach Iteration {it} von {n_snaps - 1}**")
            st.plotly_chart(build_tour(xy, run.snapshots[it], ghost=run.best_tour if it < n_snaps - 1 else None), width="stretch", key=f"s2_map_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Guided Local Search: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s3_gls")
            c2.markdown(f"**Tabu Search, gleiches Budget** – {a.tabu_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.tabu.best_tour), width="stretch", key="s3_tabu")


if auto_play:
    for s in STEP_LABELS:
        if s == 2:
            for f in _frames():
                _render(2, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, iteration)
            time.sleep(1.2)
    step = 3
elif play_search:
    for f in _frames():
        _render(2, f)
        time.sleep(0.1)
else:
    _render(step, iteration)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Nachbarn in {run.iterations} Iterationen (rund {run.evaluations // max(run.iterations, 1)} je Iteration - die volle Nachbarschaft); {run.penalizations} davon Strafen statt eines Zuges (erweitertes lokales Optimum erreicht).")
else:
    st.caption(f"Links die beste Tour von Guided Local Search, rechts die beste Tour von Tabu Search - beide aus derselben Startlösung mit demselben Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung (nur die Startlösung streut, Guided Local Search selbst ist deterministisch): Vergleiche gelten für diesen Lauf."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste Tour", f"{a.gap:.1f} %", delta=f"letzte Tour {a.final_gap:.1f} %", delta_color="off", help="Abstand zur Schranke der kürzesten je besuchten Tour; im Delta der der letzten Tour der Kette.")
m2.metric("Tabu Search, gleiches Budget", f"{a.tabu_gap:.1f} %", delta=f"{_fmt_int(a.tabu.evaluations)} Bewertungen", delta_color="off", help="Tabu Search (kalibrierte Tenure 20) aus derselben Startlösung mit demselben Budget - der direkte Vergleichspartner.")
m3.metric("HC mit Neustarts (voller Rescan)", f"{a.hcr_gap:.1f} %", delta=f"{a.hcr_starts} Abstiege, gleiches Budget", delta_color="off", help="So viele Abstiege aus zufälligen Startlösungen, wie ins Budget passen; die beste Tour zählt.")
m4.metric("Strafen statt Zug", f"{a.penalization_rate:.0%}", delta=f"{run.penalizations} von {run.iterations}", delta_color="off", help="Anteil der Iterationen, in denen (statt eines Zuges) eine Kante bestraft wurde, weil die erweiterte Suche feststeckte.")

if code == "beats_tabu":
    st.success(f"✅ Besser als Tabu Search (gleiches Budget): {a.gap:.1f} % über der Schranke gegen {a.tabu_gap:.1f} %. Die dauerhafte Strafe verwertet das Budget hier wirksamer als Tabus verfallende Sperre. Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: Guided Local Search {a.gap:.1f} %, Tabu Search {a.tabu_gap:.1f} % über der Schranke. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Tabu Search ist besser: {a.tabu_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget. Bei kleinem Budget (unter 100 Tausend) sind beide Verfahren praktisch gleichauf - siehe README 'Was nicht funktioniert hat'.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarn", "Rechenzeit"],
              "Guided Local Search (beste Tour)": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.evaluations), unit_time(a.seconds)],
              "Guided Local Search (letzte Tour)": [f"{run.final_length:.1f}", f"{a.final_gap:.2f} %", "–", "–"],
              "Tabu Search": [f"{a.tabu.best_length:.1f}", f"{a.tabu_gap:.2f} %", _fmt_int(a.tabu.evaluations), unit_time(a.tabu_seconds)],
              "HC mit Neustarts": [f"{T.tour_length(a.hcr_tour, a.D):.1f}", f"{a.hcr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.hcr_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Lambda", "Budget", "Iterationen", "Bewertungen je Iteration", "Startlösung", "Kreuzungen der besten Tour"],
              "Einstellung": [f"{settings.lam:g}", _fmt_int(settings.budget), f"{run.iterations}", f"{run.evaluations // max(run.iterations, 1)}", C.START_LABELS[settings.start], f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist ein bewerteter Nachbar, dieselbe Einheit wie in den Geschwister-Demos - aber Guided Local Search bewertet wie Tabu Search die volle Nachbarschaft auf einen Schlag, nicht einzelne Vorschläge nacheinander. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, Lambda und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 20 bis 90 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    categorical = sweep_param == "start"
    labels = {"start": C.START_LABELS}.get(sweep_param)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], categorical=categorical, key_labels=labels), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band bzw. Balken) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gepunktet: Tabu Search, gleiches Budget; gestrichelt: Hill Climbing mit Neustarts und ein einzelner Abstieg.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: wann zieht die dauerhafte Strafe an Tabu Search vorbei?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 90 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "GLS (%)": [f"{r['gap']:.2f}" for r in rows_b], "Tabu (%)": [f"{r['tabu']:.2f}" for r in rows_b],
              "HC-Neustarts (%)": [f"{r['hcr']:.2f}" for r in rows_b], "ein Abstieg (%)": [f"{r['hc']:.2f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, Lambda 5). Bei 10-50 Tausend Vorschlägen ist Guided Local Search mit Tabu Search NAHEZU IDENTISCH (beide bewerten die volle Nachbarschaft, bei so wenigen Iterationen hat die Strafe noch keine Gelegenheit zu wirken). "
               "Ab 100 Tausend trennen sich die Kurven: bei 100 Tausend liegt Tabu Search noch knapp vorn, ab 200 Tausend zieht Guided Local Search klar davon und der Abstand zur Schranke schrumpft mit jeder Budget-Verdopplung stärker als bei Tabu Search (2 Millionen: 0.52 % gegen 1.9 %) - die dauerhafte Strafe verwertet zusätzliches Budget offenbar wirksamer als eine verfallende Sperre.")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 20 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten, 20 Tabu-Search-Läufe und 20 Abstiege..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["gls"], sp["tabu"], sp["hc"]), width="stretch", key="spread_chart")
    s1, s2, s3 = st.columns(3)
    s1.metric("Guided Local Search: Mittel ± Streuung", f"{sp['gls'].mean():.2f} ± {sp['gls'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten (nur die Startlösung streut).")
    s2.metric("Tabu Search: Mittel ± Streuung", f"{sp['tabu'].mean():.2f} ± {sp['tabu'].std():.2f} %", help="Gleiches Budget, gleiche 20 Startlösungen.")
    s3.metric("Ein Hill-Climbing-Abstieg: Mittel ± Streuung", f"{sp['hc'].mean():.2f} ± {sp['hc'].std():.2f} %", help="Ein Abstieg je Kette aus derselben zufälligen Startlösung.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (nur die Startlösung wechselt - beide Suchen sind selbst deterministisch).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 90 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget). Bei einem für 60 Stopps kalibrierten Budget reicht das Budget ab rund 100-150 Stopps für KEINES der beiden Verfahren mehr aus, um über die ersten Züge hinauszukommen - "
               "Guided Local Search und Tabu Search liegen dort auf zwei Nachkommastellen genau gleichauf (derselbe geteilte Budget-Engpass, nicht ein Unterschied im Mechanismus). Bei mittlerer Instanzgröße (40-60 Stopps) schlägt Guided Local Search Tabu Search dagegen klar.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Budget reicht für genug teure Iterationen** | Bei 25 Tausend Vorschlägen (60 Stopps, nur ~15 Iterationen): **155 %** über der Schranke - praktisch identisch mit Tabu Search bei gleichem Budget. Die Strafe braucht Iterationen, um überhaupt zu wirken. | **ILS/VNS** (viele billige Iterationen statt weniger teurer) |
| **Lambda passt zur Instanz** | Zu schwache Strafe (0.1): **7.74 %**, kaum besser als ein Abstieg (7.85 %) - ohne wirksamen Druck bleibt die Suche im ersten lokalen Optimum hängen. Zu starke Strafe (30): **5.69 %**, schlechter als Lambda 5 (3.39 %) - Vermeidung dominiert über Kürze. | Kein direkter Nachfolger; dieselbe Lehre wie SA's Temperatur, ILS' Störstärke, VNS' k_max, Tabus Tenure, GRASPs RCL-Größe |
| **Jede Iteration ist bezahlbar** | Bei 200 Stopps kostet eine Iteration rund 20 Tausend Bewertungen - 1 Million Vorschläge reichen nur für ~50 Iterationen (**307 %** über der Schranke, exakt wie Tabu Search bei gleichem Budget). Skaliert nicht besser als Tabu Search - derselbe Preis für die volle Nachbarschaft. | **Kandidatenliste + Don't-Look-Bits** (aus der Hill-Climbing-Demo) |
| **Die Strafe darf sich über viele lokale Optima hinweg ansammeln** | Bei sehr kleinem Budget kommt die Suche über das erste lokale Optimum kaum hinaus - die Strafe hat noch keine Gelegenheit, mehrfach zu wirken. Genau dort ist der Vorteil gegenüber Tabu Search auch messbar null. | (kein Nachfolger nötig - dieselbe Lehre wie Tabus eigener Budget-Boden) |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie: Simulated Annealing (Zufall statt Gedächtnis), Iterated Local Search/Variable "
    "Neighborhood Search (gezielte Störung + Wiederabstieg), GRASP (randomisierte Konstruktion) sind andere Antworten auf dieselbe "
    "Schwäche der Wurzel; der Nachbarschafts-Zweig (Lin-Kernighan, VLSN, VRP-Nachbarschaften) ändert stattdessen die Nachbarschaft selbst."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Zug.** Ein 2-opt-Zug $(i, j)$ entfernt die Kanten $(t_i, t_{i+1})$ und $(t_j, t_{j+1})$, fügt $(t_i, t_j)$ und $(t_{i+1}, t_{j+1})$ ein (Stück zwischen $i+1$ und $j$ umgekehrt).

**Strafen.** $p(u, v) \geq 0$ = Strafzahl der Kante $\{u, v\}$, anfangs 0, verfällt nie. Erweiterte Zielfunktion: $\tilde{L}(\pi) = L(\pi) + \lambda \sum_{\{u,v\} \in \pi} p(u, v)$.

**Zugauswahl.** In jeder Iteration: $\pi' = \arg\min_{(i,j) \text{ gültig}} \tilde{L}(\pi \text{ mit Zug } (i,j))$. Verbessert kein Zug $\tilde{L}$ (erweitertes lokales Optimum), wird stattdessen die Kante mit dem höchsten Nutzen $\text{util}(u,v) = d_{uv} / (1 + p(u,v))$ unter den aktuellen Tourkanten bestraft: $p(u,v) \leftarrow p(u,v) + 1$.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgrößen bei gleichem Budget: Tabu Search (kalibrierte Tenure 20), ein Hill-Climbing-Abstieg und Hill Climbing mit Neustarts (voller Rescan).

**Grenzen.** (1) Eine Iteration kostet $O(n^2)$ Bewertungen wie bei Tabu Search - dasselbe hohe Mindestbudget. (2) Lambda ist ein Sweet-Spot-Parameter, kein "mehr ist sicherer". (3) Die Strafe braucht mehrere lokale Optima, um zu wirken - bei sehr kleinem Budget bringt sie nichts.

Implementiert in `gls_algorithm.py` (die Suchschleife: erweiterte Zielfunktion, Nutzen-Bestrafung), `gls_tour.py` (Nachbarschaften, Abstieg, Schranke - aus der Hill-Climbing-Demo), `gls_tabu_algorithm.py` (Tabu Search als Vergleichspartner, byte-identisch zur tabu-search-demo), `gls_scenario.py` (Instanzen), `gls_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Trajektorien-Metaheuristiken: HC bis ALNS](https://sebastianhanisch.net/konzepte-trajektorien-metaheuristiken.html)."
)
