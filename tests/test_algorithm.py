"""gls_algorithm.run: Budget-Buchführung (volle Nachbarschaft je Iteration), Strafmatrix-Verwaltung (verfällt NIE, anders als
Tabus Sperrfrist), Nutzen-Bestrafung bei einem erweiterten lokalen Optimum, Regressionsschutz, Determinismus, "Lambda=0"-Ablation
(die Suche friert nach dem ersten lokalen Optimum ein, statt zu pendeln wie Tabu Search mit Tenure 0)."""

import numpy as np
import pytest

import gls_algorithm as GLS
import gls_tabu_algorithm as TABU
import gls_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


def test_tour_stays_valid_and_length_matches_recomputed_length():
    D = _instance(20, 1)
    start = T.random_tour(20, np.random.default_rng(0))
    r = GLS.run(D, start, lam=0.5, budget=20000)
    assert sorted(r.best_tour.tolist()) == list(range(20))
    assert sorted(r.final_tour.tolist()) == list(range(20))
    assert r.best_length == pytest.approx(T.tour_length(r.best_tour, D), abs=1e-6)
    assert r.final_length == pytest.approx(T.tour_length(r.final_tour, D), abs=1e-6)


def test_budget_is_respected_and_exceeded_by_at_most_one_full_neighborhood():
    D = _instance(25, 2)
    start = T.random_tour(25, np.random.default_rng(1))
    budget = 20000
    r = GLS.run(D, start, lam=0.5, budget=budget)
    n = len(D)
    max_neighborhood = n * (n - 3) // 2 + n                  # grosszuegige obere Schranke fuer eine volle 2-opt-Nachbarschaft
    assert r.evaluations >= budget
    assert r.evaluations <= budget + max_neighborhood
    assert r.iterations >= 1


def test_best_length_is_never_worse_than_the_start():
    D = _instance(30, 3)
    start = T.random_tour(30, np.random.default_rng(2))
    start_length = T.tour_length(start, D)
    r = GLS.run(D, start, lam=1.0, budget=30000)
    assert r.best_length <= start_length + 1e-6


def test_the_search_takes_the_best_augmented_move_an_independent_replay_confirms_it():
    """Unabhaengige Nachrechnung: aus dem debug_trace (i, j, war_strafe) wird die Strafmatrix NUR anhand der oeffentlichen
    Regel ("bei einem erweiterten lokalen Optimum wird die Kante mit dem hoechsten Nutzen unter den aktuellen Tourkanten
    bestraft") neu aufgebaut - unabhaengig von der internen `penalty`-Datenstruktur des Algorithmus. Bei jedem Zug muss
    er der beste unter der so rekonstruierten erweiterten Zielfunktion sein; bei jeder Strafe muss die bestrafte Kante
    tatsaechlich die hoechste Nutzen-Kante der aktuellen Tour sein."""
    D = _instance(16, 4)
    start = T.random_tour(16, np.random.default_rng(3))
    lam = 2.0
    r = GLS.run(D, start, lam=lam, budget=8000, keep_snapshots=True, debug_trace=True)
    n = len(D)
    penalty = {}                                                # (min(u,v), max(u,v)) -> Strafzahl, verfaellt nie
    moves_checked = penalties_checked = 0
    for it, (a, b, was_penalty) in enumerate(r.debug[:200]):     # die ersten 200 Eintraege genuegen fuer eine gruendliche Pruefung
        cur = r.snapshots[it]
        delta, valid = T._delta_2opt(cur, D)
        n_cur = len(cur)
        nxt = np.roll(cur, -1)
        pen_lookup = np.zeros((n, n))
        for (u, v), p in penalty.items():
            pen_lookup[u, v] = pen_lookup[v, u] = p
        e = pen_lookup[cur, nxt]
        delta_pen = pen_lookup[np.ix_(cur, cur)] + pen_lookup[np.ix_(nxt, nxt)] - e[:, None] - e[None, :]
        augmented = delta + lam * delta_pen
        masked = np.where(valid, augmented, np.inf)
        best_augmented = float(np.min(masked))
        if was_penalty:
            assert best_augmented >= -1e-9                       # kein Zug haette die erweiterte Zielfunktion verbessert
            edges = T.tour_edges(cur)
            edge_arr = np.array(sorted(edges))
            lengths = D[edge_arr[:, 0], edge_arr[:, 1]]
            pens = np.array([penalty.get((min(int(u2), int(v2)), max(int(u2), int(v2))), 0) for u2, v2 in edge_arr])
            utility = lengths / (1.0 + pens)
            expected_u, expected_v = edge_arr[int(np.argmax(utility))]
            assert {int(a), int(b)} == {int(expected_u), int(expected_v)}
            key = (min(int(a), int(b)), max(int(a), int(b)))
            penalty[key] = penalty.get(key, 0) + 1
            penalties_checked += 1
        else:
            i, j = a, b
            assert masked[i, j] <= best_augmented + 1e-6
            moves_checked += 1
    assert moves_checked > 10 and penalties_checked > 5           # der Test hat tatsaechlich beide Zweige durchlaufen


def test_zero_lambda_freezes_after_reaching_a_local_optimum():
    """Lambda=0: die Strafe hat kein Gewicht in der erweiterten Zielfunktion, ist die Suche einmal in einem lokalen
    Optimum, kann sie das nie mehr verlassen (anders als Tabu Search mit Tenure 0, das zwischen zwei Touren pendelt) -
    jede weitere 'Iteration' ist eine wirkungslose Strafe, die Tour aendert sich nicht mehr."""
    D = _instance(14, 6)
    start = T.random_tour(14, np.random.default_rng(5))
    r = GLS.run(D, start, lam=0.0, budget=4000, keep_snapshots=True)
    tail = [tuple(np.asarray(s).tolist()) for s in r.snapshots[-20:]]
    assert len(set(tail)) == 1                                   # eingefroren auf genau einer Tour


def test_deterministic_given_the_same_start_and_different_for_another_start():
    """Guided Local Search selbst ist deterministisch (kein Zufall im Kern) - derselbe Lauf mit derselben Startloesung
    muss IMMER dasselbe Ergebnis liefern; nur die Startloesung (die Kette in der Auswertungsschicht) entscheidet."""
    D = _instance(25, 7)
    start_a = T.random_tour(25, np.random.default_rng(6))
    start_c = T.random_tour(25, np.random.default_rng(20))
    a = GLS.run(D, start_a, lam=0.5, budget=10000)
    b = GLS.run(D, start_a, lam=0.5, budget=10000)
    c = GLS.run(D, start_c, lam=0.5, budget=10000)
    assert np.array_equal(a.best_tour, b.best_tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.best_tour, c.best_tour) or a.evaluations != c.evaluations


def test_negative_lambda_is_rejected():
    D = _instance(10, 0)
    start = T.random_tour(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        GLS.run(D, start, lam=-1.0, budget=1000)


def test_snapshots_are_only_kept_when_requested():
    D = _instance(12, 0)
    start = T.random_tour(12, np.random.default_rng(0))
    with_snaps = GLS.run(D, start, lam=0.5, budget=3000, keep_snapshots=True)
    without = GLS.run(D, start, lam=0.5, budget=3000, keep_snapshots=False)
    assert len(with_snaps.snapshots) == with_snaps.iterations + 1
    assert without.snapshots == []


def test_penalty_never_decreases():
    """Anders als Tabus Sperrfrist verfaellt eine Strafe nie - die Zahl der Strafen kann nur wachsen, nie sinken."""
    D = _instance(18, 11)
    start = T.random_tour(18, np.random.default_rng(4))
    r = GLS.run(D, start, lam=1.0, budget=6000, debug_trace=True)
    assert r.penalizations >= 0
    assert r.penalizations == sum(1 for (_, _, was_penalty) in r.debug if was_penalty)


def _expected_trace_axis(per_iteration, budget, trace_points=300):
    """Unabhängig aus der Definition: Verlaufspunkt bei Bewertungsstand 0, danach in der ersten Iteration, die die nächste Vielfachen-Schwelle
    (alle budget // trace_points BEWERTETE Nachbarn) erreicht, und in der letzten Iteration."""
    every = max(1, budget // trace_points)
    xs, ev, threshold = [0], 0, every
    while ev < budget:
        ev += per_iteration
        if ev >= threshold or ev >= budget:
            xs.append(ev)
            threshold = (ev // every + 1) * every
    return xs


@pytest.mark.parametrize("module, kwargs", [(GLS, {"lam": 0.5}), (TABU, {"tenure": 5})])
@pytest.mark.parametrize("n, budget", [(10, 30000), (20, 30000), (40, 100000)])
def test_the_trace_is_recorded_per_evaluated_neighbours_not_per_iteration(module, kwargs, n, budget):
    """Die Achse heißt "Bewertete Nachbarn": der Verlauf muss alle budget // 300 BEWERTUNGEN einen Punkt tragen. Alle 333 ITERATIONEN
    (alter Stand) blieben bei n = 40 und Budget 100000 nur 2 Punkte statt rund 100 übrig."""
    D = _instance(n, 7)
    start = T.random_tour(n, np.random.default_rng(3))
    r = module.run(D, start, budget=budget, keep_snapshots=False, **kwargs)
    per_iteration = r.evaluations // r.iterations
    assert per_iteration * r.iterations == r.evaluations                 # jede Iteration bewertet gleich viele Paare
    assert r.trace_iter.tolist() == _expected_trace_axis(per_iteration, budget)
    assert len(r.trace_length) == len(r.trace_best) == len(r.trace_iter) and r.trace_iter[-1] == r.evaluations
    assert len(r.trace_iter) >= min(r.iterations + 1, 150)               # nicht nur 2 bis 5 Punkte
    assert np.all(np.diff(r.trace_best) <= 1e-9)                         # beste Länge fällt nur
