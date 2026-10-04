"""Orakel-Test für den Hill-Climbing-Kern (`gls_tour`): Nachbarschafts-Deltas, Auswahlregeln, Bewertungszähler, Kreuzungszählung und 1-Baum-Schranke
gegen einen ANDEREN Rechenweg - Tourlängen werden hier durch vollständiges Neuberechnen nach dem Zerschneiden/Umbauen der Tourliste bestimmt
(nicht über die Delta-Formeln der Matrizen), Kreuzungen exakt mit Brüchen, die Schranke gegen die exakte Held-Karp-DP (kein Solver nötig).
Kein Test vergleicht den Code mit sich selbst oder mit eingefrorenen Werten."""

import itertools
from fractions import Fraction

import numpy as np

import gls_tour as T

EPS = 1e-9


def _len(tour, Dl):
    n = len(tour)
    return sum(Dl[tour[k]][tour[(k + 1) % n]] for k in range(n))


def _neighbors(t, Dl, neighborhood):
    """(Art, neue Tour) aller Nachbarn in der dokumentierten Reihenfolge (swap/2opt: nach (i, j); or-opt: Länge, Stück i, Kante m der Tour)."""
    t = [int(x) for x in t]
    n = len(t)
    out = []
    if neighborhood == "swap":
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u = t[:]
                u[i], u[j] = u[j], u[i]
                out.append(("swap", u))
    if neighborhood in ("2opt", "2opt+oropt"):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                out.append(("2opt", t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]))
    if neighborhood in ("oropt", "2opt+oropt"):
        for length in (1, 2, 3):
            if n < length + 3:
                continue
            for i in range(n):
                seg = [t[(i + k) % n] for k in range(length)]
                rest = [t[(i + length + k) % n] for k in range(n - length)]
                for q in sorted(range(len(rest) - 1), key=lambda q: (i + length + q) % n):
                    best = min(((_len(rest[:q + 1] + s + rest[q + 1:], Dl), rest[:q + 1] + s + rest[q + 1:]) for s in (seg, seg[::-1])), key=lambda x: x[0])
                    out.append((f"oropt{length}", best[1]))
    return out


def _instances(rng, trials, n_lo, n_hi):
    for _ in range(trials):
        n = int(rng.integers(n_lo, n_hi))
        # Zufallspunkte und Gitterpunkte (viele Gleichstände, doppelte Punkte)
        xy = rng.random((n, 2)) * 100 if rng.random() < 0.6 else rng.integers(0, 4, (n, 2)).astype(float)
        yield n, xy, T.dist_matrix(xy)


def test_neighborhood_deltas_selection_rules_and_evaluation_counts_match_a_full_recomputation_oracle():
    rng = np.random.default_rng(2024)
    for n, xy, D in _instances(rng, 45, 4, 10):
        Dl = D.tolist()
        t = rng.permutation(n)
        base = T.tour_length(t, D)
        for neighborhood in T.NEIGHBORHOODS:
            orc = _neighbors(t, Dl, neighborhood)
            od = sorted(_len(u, Dl) - base for _, u in orc)
            assert np.allclose(od, sorted(T.neighbor_deltas(t, D, neighborhood).tolist()), atol=1e-7), (neighborhood, n)
            move, count = T.find_move(t, D, neighborhood, "best")
            assert count == len(orc)
            if od and od[0] < -EPS:
                assert move is not None and abs(move[-1] - od[0]) < 1e-7
                new = T.apply_move(t, move)
                assert abs(_len(new.tolist(), Dl) - (base + move[-1])) < 1e-7
            else:
                assert move is None
            move, count = T.find_move(t, D, neighborhood, "first")
            first = next((k for k, (_, u) in enumerate(orc) if _len(u, Dl) - base < -EPS), None)
            if first is None:
                assert move is None and count == len(orc)
            else:
                assert move is not None and count == first + 1
                assert abs(move[-1] - (_len(orc[first][1], Dl) - base)) < 1e-7
                assert abs(_len(T.apply_move(t, move).tolist(), Dl) - (base + move[-1])) < 1e-7


def test_descent_ends_in_a_local_optimum_found_by_full_recomputation():
    rng = np.random.default_rng(7)
    for n, xy, D in _instances(rng, 12, 5, 10):
        Dl = D.tolist()
        for neighborhood in T.NEIGHBORHOODS:
            for rule in T.RULES:
                r = T.descend(D, rng.permutation(n), neighborhood, rule)
                length = _len(r.tour.tolist(), Dl)
                assert abs(r.length - length) < 1e-7
                assert all(_len(u, Dl) >= length - 1e-7 for _, u in _neighbors(r.tour, Dl, neighborhood))


def _proper_crossing(p, q, r, s):
    """Echtes Schneiden zweier Strecken: p + a (q-p) = r + b (s-r) mit 0 < a, b < 1, exakt in Brüchen."""
    px, py = map(Fraction, p)
    qx, qy = map(Fraction, q)
    rx, ry = map(Fraction, r)
    sx, sy = map(Fraction, s)
    d1x, d1y, d2x, d2y = qx - px, qy - py, sx - rx, sy - ry
    den = d1x * d2y - d1y * d2x
    if den == 0:
        return False
    a = ((rx - px) * d2y - (ry - py) * d2x) / den
    b = ((rx - px) * d1y - (ry - py) * d1x) / den
    return 0 < a < 1 and 0 < b < 1


def test_count_crossings_matches_an_exact_fraction_oracle():
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(4, 12))
        xy = rng.integers(0, 6, (n, 2)).astype(float) if rng.random() < 0.6 else rng.random((n, 2))
        t = rng.permutation(n)
        expected = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b, c, d = t[i], t[(i + 1) % n], t[j], t[(j + 1) % n]
                if len({a, b, c, d}) == 4 and _proper_crossing(xy[a], xy[b], xy[c], xy[d]):
                    expected += 1
        assert T.count_crossings(xy, t) == expected
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    assert T.count_crossings(bowtie, [0, 1, 2, 3]) == 1 and T.count_crossings(bowtie, [0, 2, 1, 3]) == 0


def _held_karp_optimum(Dl):
    """Exakte kürzeste Rundtour (Held-Karp-DP über Teilmengen), n <= 9."""
    n = len(Dl)
    if n == 1:
        return 0.0
    if n == 2:
        return 2 * Dl[0][1]
    m = n - 1
    dp = {(1 << j, j): Dl[0][j + 1] for j in range(m)}
    for mask in range(1, 1 << m):
        for j in range(m):
            cur = dp.get((mask, j))
            if cur is None:
                continue
            for k in range(m):
                if not (mask >> k) & 1:
                    key = (mask | 1 << k, k)
                    v = cur + Dl[j + 1][k + 1]
                    if v < dp.get(key, float("inf")):
                        dp[key] = v
    return min(dp[((1 << m) - 1, j)] + Dl[j + 1][0] for j in range(m))


def test_dynamic_programming_oracle_agrees_with_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(10):
        n = int(rng.integers(3, 8))
        Dl = T.dist_matrix(rng.random((n, 2)) * 50).tolist()
        brute = min(_len([0, *p], Dl) for p in itertools.permutations(range(1, n)))
        assert abs(_held_karp_optimum(Dl) - brute) < 1e-9


def test_one_tree_bound_never_exceeds_the_exact_optimum_and_degenerate_sizes_are_handled():
    rng = np.random.default_rng(5)
    for n, xy, D in _instances(rng, 40, 1, 9):
        Dl = D.tolist()
        opt = _held_karp_optimum(Dl)
        upper = T.descend(D, np.arange(n), "2opt+oropt", "best", keep_steps=False).length if n > 3 else opt
        assert T.held_karp_bound(D, upper, 300) <= opt + 1e-7


# --- Guided Local Search und Tabu Search: vollständige Nachsimulation auf Tourlisten ---------------------------------------------------------------
# Jede Iteration wird hier aus der DEFINITION neu gerechnet: alle (i, j)-2-opt-Nachbarn als explizite Tourlisten, erweiterte Kosten = Länge + lam * Summe der
# Strafzahlen der Kanten der neuen Tour (Strafen als Dict über sortierte Kantenpaare), gewählt wird das Minimum (Gleichstand: kleinstes (i, j), wie
# die zeilenweise argmin-Auswahl der dokumentierten Reihenfolge); bei Strafe wird die Kante mit maximalem Länge/(1 + Strafe) bestraft (Gleichstand: lexikographisch
# kleinste Kante). Rundungs-Gleichstände (Differenz < 1e-9 zwischen bestem und zweitbestem Wert) beenden die Nachsimulation vorzeitig.


import pytest

import gls_algorithm as GLS
import gls_tabu_algorithm as TABU


def _edges(t):
    return [tuple(sorted((t[k], t[(k + 1) % len(t)]))) for k in range(len(t))]


def _neighbour_list(t):
    n = len(t)
    return [((i, j), t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]) for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)]


def _gls_resimulate(Dl, start, lam, iterations):
    t = [int(x) for x in start]
    pen = {}
    log = []
    for _ in range(iterations):
        scored = [(_len(u, Dl) + lam * sum(pen.get(e, 0) for e in _edges(u)), ij, u) for ij, u in _neighbour_list(t)]
        cur = _len(t, Dl) + lam * sum(pen.get(e, 0) for e in _edges(t))
        values = sorted(s[0] for s in scored)
        if (len(values) > 1 and abs(values[0] - values[1]) < 1e-9) or abs(values[0] - cur) < 1e-9:
            return log, False
        best = min(scored, key=lambda s: (s[0], s[1]))
        if best[0] < cur - 1e-9:
            log.append(("move", best[1]))
            t = best[2]
        else:
            utils = [(Dl[a][b] / (1 + pen.get((a, b), 0)), (a, b)) for (a, b) in sorted(set(_edges(t)))]
            top = max(u for u, _ in utils)
            if sum(abs(u - top) < 1e-9 for u, _ in utils) > 1:
                return log, False
            edge = max(utils, key=lambda x: x[0])[1]
            pen[edge] = pen.get(edge, 0) + 1
            log.append(("penalty", edge))
    return log, True


def test_guided_local_search_equals_a_full_resimulation_from_the_definition():
    rng = np.random.default_rng(55)
    complete = moves = penalties = 0
    for _ in range(120):
        n = int(rng.integers(5, 10))
        xy = rng.random((n, 2)) * 100
        D = T.dist_matrix(xy)
        start = rng.permutation(n)
        lam = float(rng.choice([0.0, 0.3, 1.0, 5.0, 30.0]))
        its = int(rng.integers(5, 40))
        # Budget so wählen, dass genau `its` Iterationen laufen: Bewertungen je Iteration = Zahl der gültigen Paare
        per = len(_neighbour_list(list(range(n))))
        run = GLS.run(D, start, lam=lam, budget=per * its - 1, keep_snapshots=False, debug_trace=True)
        assert run.iterations == its and run.evaluations == per * its
        log, ok = _gls_resimulate(D.tolist(), start, lam, its)
        k = len(log)
        shown = [("penalty", (min(a, b), max(a, b))) if pen else ("move", (a, b)) for a, b, pen in run.debug[:k]]
        assert shown == log
        if ok:
            complete += 1
            moves += sum(x[0] == "move" for x in log)
            penalties += sum(x[0] == "penalty" for x in log)
            assert run.penalizations == sum(1 for _, _, p in run.debug if p) == sum(x[0] == "penalty" for x in log)
    assert complete >= 60 and moves > 100 and penalties > 50


def _tabu_resimulate(Dl, start, tenure, iterations):
    """Tabu Search aus der Definition: immer der beste erlaubte Nachbar; verboten ist ein Zug, der eine in den letzten `tenure` Iterationen ENTFERNTE Kante
    wieder einfügt - außer er liefert eine neue beste Tour (Aspiration); sind alle verboten, gilt der beste aller Züge (Notfall)."""
    t = [int(x) for x in start]
    best = _len(t, Dl)
    removed_at = {}                                               # Kante -> Nummer (ab 1) der Iteration, in der sie zuletzt entfernt wurde
    log = []
    for it in range(1, iterations + 1):
        scored = [(_len(u, Dl), ij, u) for ij, u in _neighbour_list(t)]

        def tabu(u):
            return any(e in removed_at and it - removed_at[e] <= tenure for e in set(_edges(u)) - set(_edges(t)))

        allowed = [s for s in scored if not tabu(s[2]) or s[0] < best - 1e-9]
        pool = allowed or scored
        ranked = sorted(s[0] for s in pool)
        if len(ranked) > 1 and abs(ranked[0] - ranked[1]) < 1e-9:
            return log, False
        if any(abs(s[0] - best) < 1e-9 for s in pool):
            return log, False
        chosen = min(pool, key=lambda s: (s[0], s[1]))
        for e in set(_edges(t)) - set(_edges(chosen[2])):
            removed_at[e] = it
        t = chosen[2]
        best = min(best, chosen[0])
        log.append((chosen[1], not allowed))
    return log, True


def test_tabu_search_equals_a_full_resimulation_from_the_definition():
    rng = np.random.default_rng(66)
    compared = overrides = 0
    for _ in range(120):
        n = int(rng.integers(5, 10))
        xy = rng.random((n, 2)) * 100
        D = T.dist_matrix(xy)
        start = rng.permutation(n)
        tenure = int(rng.choice([0, 1, 3, 8, 20]))
        its = int(rng.integers(5, 40))
        per = len(_neighbour_list(list(range(n))))
        run = TABU.run(D, start, tenure=tenure, budget=per * its - 1, keep_snapshots=False, debug_trace=True)
        assert run.iterations == its and run.evaluations == per * its
        log, ok = _tabu_resimulate(D.tolist(), start, tenure, its)
        assert [(i, j, o) for (i, j), o in log] == run.debug[:len(log)]
        compared += len(log)
        if ok:
            overrides += sum(o for _, o in log)
            assert run.overrides == sum(o for _, o in log)
    assert compared > 800 and overrides > 10               # Gleichstände zum Rekordwert (Aspiration) beenden einen Vergleich vorzeitig, vorher stimmt jeder Zug
