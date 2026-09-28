"""Tabu Search (Glover 1986/1989-90) für eine Rundtour (TSP): wie ein Hill-Climbing-Abstieg mit der Regel "beste Verbesserung" (die
volle n×n-Nachbarschaft wird jede Iteration neu bewertet), aber es wird IMMER der beste verfügbare Zug ausgeführt - auch wenn er die
Tour verlängert. Damit die Suche nicht sofort zwischen zwei Touren hin- und herpendelt, werden die beiden von einem Zug ENTFERNTEN
Kanten für `tenure` Iterationen tabu: ein künftiger Zug, der eine davon wieder herstellen würde, ist verboten - es sei denn, er
verbessert die beste je gefundene Tour (Aspirationskriterium). Sind ausnahmsweise alle gültigen Züge tabu, wird der beste davon
trotzdem ausgeführt (Notfall-Überschreibung, sonst bliebe die Suche stecken).

Byte-identisch mit tabu_algorithm.py aus der Tabu-Search-Demo (bis auf den Vehikel-Import) - dient hier als Vergleichspartner für
Guided Local Search, nicht als eigenständiges Stück."""

from dataclasses import dataclass, field

import numpy as np

import gls_tour as T


@dataclass
class Run:
    best_tour: np.ndarray
    best_length: float
    final_tour: np.ndarray
    final_length: float
    evaluations: int = 0
    iterations: int = 0
    overrides: int = 0                                      # Iterationen, in denen ALLE gültigen Züge tabu waren (Notfall-Überschreibung)
    snapshots: list = field(default_factory=list)            # aktuelle Tour nach jeder Iteration (nur bei keep_snapshots=True)
    trace_iter: np.ndarray = None
    trace_length: np.ndarray = None                          # aktuelle Länge über die Iterationen
    trace_best: np.ndarray = None
    tenure: int = 20
    debug: list = field(default_factory=list)                # nur bei debug_trace=True: (i, j, war_override) je Iteration - fürs Testen


def run(D, start, tenure=20, budget=100000, keep_snapshots=True, trace_points=300, debug_trace=False):
    """Ein Lauf. `tenure`: wie viele Iterationen eine entfernte Kante verboten bleibt. `budget`: Zahl der bewerteten
    Nachbarschaften insgesamt (eine Iteration bewertet die volle 2-opt-Nachbarschaft auf einen Schlag).
    Kein Zufall im Kern (immer der beste erlaubte Zug) - kein Seed-Parameter nötig; nur die übergebene Startlösung entscheidet.
    `debug_trace=True` (nur für Tests): sammelt je Iteration `(i, j, war_override)`."""
    if tenure < 0:
        raise ValueError(tenure)
    n = len(D)
    t = np.asarray(start, dtype=np.int64).copy()
    length = T.tour_length(t, D)
    best_tour, best_length = t.copy(), length
    tabu_until = np.zeros((n, n), dtype=np.int64)             # tabu_until[u, v] > it  <=>  Kante (u, v) ist noch tabu
    evaluations = iterations = overrides = 0
    snapshots = [t.copy()] if keep_snapshots else []
    debug = []
    trace_every = max(1, budget // trace_points)
    tr_it, tr_len, tr_best = [0], [length], [length]

    while evaluations < budget:
        delta, valid = T._delta_2opt(t, D)
        nxt = np.roll(t, -1)
        tabu0 = tabu_until[np.ix_(t, t)] > iterations
        tabu1 = tabu_until[np.ix_(nxt, nxt)] > iterations
        is_tabu = tabu0 | tabu1
        candidate_length = length + delta
        aspiring = candidate_length < best_length - 1e-9
        allowed = valid & (~is_tabu | aspiring)
        override = not allowed.any()
        mask = valid if override else allowed
        masked = np.where(mask, delta, np.inf)
        k = int(np.argmin(masked))
        i, j = divmod(k, n)
        evaluations += int(valid.sum())
        iterations += 1
        if override:
            overrides += 1
        if debug_trace:
            debug.append((i, j, override))

        removed_a, removed_b = (t[i], t[i + 1]), (t[j], t[(j + 1) % n])
        length += float(delta[i, j])
        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        tabu_until[removed_a[0], removed_a[1]] = tabu_until[removed_a[1], removed_a[0]] = iterations + tenure
        tabu_until[removed_b[0], removed_b[1]] = tabu_until[removed_b[1], removed_b[0]] = iterations + tenure
        if length < best_length - 1e-9:
            best_length, best_tour = length, t.copy()
        if keep_snapshots:
            snapshots.append(t.copy())
        if iterations % trace_every == 0 or evaluations >= budget:
            tr_it.append(evaluations)
            tr_len.append(length)
            tr_best.append(best_length)

    final_length = T.tour_length(t, D)                        # Rundungsfehler der Delta-Summen beseitigen
    best_length = T.tour_length(best_tour, D)
    return Run(best_tour, best_length, t, final_length, evaluations, iterations, overrides, snapshots,
               np.array(tr_it), np.array(tr_len), np.array(tr_best), tenure, debug)
