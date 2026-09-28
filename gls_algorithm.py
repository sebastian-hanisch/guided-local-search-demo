"""Guided Local Search (Voudouris & Tsang 1999) für eine Rundtour (TSP): wie ein Hill-Climbing-Abstieg mit der Regel "beste Verbesserung"
(die volle n×n-Nachbarschaft wird jede Iteration neu bewertet), aber die Zielfunktion selbst wird um Strafen ergänzt: jede Tourkante hat
eine Strafzahl (anfangs 0, wächst nur, verfällt NIE - anders als Tabu Searchs Sperrfrist), die lokale Suche sucht auf der ERWEITERTEN
Kostenfunktion `laenge + lambda * strafen`. Findet die erweiterte Suche keinen verbessernden Zug mehr (ein erweitertes lokales Optimum),
wird die Kante mit dem höchsten Nutzen (Länge / (1 + Strafe)) unter den aktuellen Tourkanten bestraft (Strafzahl +1) - die erweiterte
Zielfunktion ändert sich dadurch, und die Suche findet einen neuen verbessernden Zug (oder bestraft erneut). Die BESTE Tour wird immer
anhand der ECHTEN (unveränderten) Länge verfolgt, nicht der erweiterten - die Erweiterung ist nur ein Führungsinstrument.

Eine Bewertung = ein geprüftes Kandidatenpaar (dieselbe Einheit wie in der ganzen Trajektorien-Metaheuristiken-Linie); eine Iteration
verbraucht wie bei Tabu Search auf einen Schlag rund n²/2 Bewertungen (die volle 2-opt-Nachbarschaft) - direkt vergleichbar bei gleichem
Budget."""

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
    penalizations: int = 0                                    # Iterationen, in denen (statt eines Zuges) eine Kante bestraft wurde
    snapshots: list = field(default_factory=list)              # aktuelle Tour nach jeder Iteration (nur bei keep_snapshots=True)
    trace_iter: np.ndarray = None
    trace_length: np.ndarray = None                            # aktuelle (ECHTE) Länge über die Iterationen
    trace_best: np.ndarray = None
    lam: float = 0.3
    debug: list = field(default_factory=list)                  # nur bei debug_trace=True: (i, j, war_strafe) je Iteration - fürs Testen


def _delta_penalty(t, P):
    """Änderung der Gesamtstrafe für jeden moeglichen 2-opt-Zug - exakt dieselbe Matrix-Konstruktion wie T._delta_2opt, nur mit der
    Strafmatrix P statt der Distanzmatrix D (zwei entfernte, zwei neue Kanten je Zug)."""
    n = len(t)
    nxt = np.roll(t, -1)
    e = P[t, nxt]
    return P[np.ix_(t, t)] + P[np.ix_(nxt, nxt)] - e[:, None] - e[None, :]


def run(D, start, lam=0.3, budget=100000, keep_snapshots=True, trace_points=300, debug_trace=False):
    """Ein Lauf. `lam`: Gewicht der Strafen in der erweiterten Zielfunktion. `budget`: Zahl der bewerteten Nachbarschaften insgesamt
    (eine Iteration bewertet die volle 2-opt-Nachbarschaft auf einen Schlag, wie bei Tabu Search).
    Anders als der Rest der Trajektorien-Metaheuristiken-Linie hat Guided Local Search KEINEN Zufall im Kern (immer der beste Zug
    unter der erweiterten Zielfunktion, keine zufällige Auswahl) - kein Seed-Parameter nötig; nur die übergebene Startlösung entscheidet.
    `debug_trace=True` (nur für Tests): sammelt je Iteration `(i, j, war_strafe)`."""
    if lam < 0:
        raise ValueError(lam)
    n = len(D)
    t = np.asarray(start, dtype=np.int64).copy()
    length = T.tour_length(t, D)
    best_tour, best_length = t.copy(), length
    penalty = np.zeros((n, n), dtype=np.int64)                 # persistente Strafzahl je Kante (u, v) - verfaellt nie
    evaluations = iterations = penalizations = 0
    snapshots = [t.copy()] if keep_snapshots else []
    debug = []
    trace_every = max(1, budget // trace_points)
    tr_it, tr_len, tr_best = [0], [length], [length]

    while evaluations < budget:
        delta, valid = T._delta_2opt(t, D)
        delta_pen = _delta_penalty(t, penalty)
        augmented = delta + lam * delta_pen
        masked = np.where(valid, augmented, np.inf)
        k = int(np.argmin(masked))
        i, j = divmod(k, n)
        evaluations += int(valid.sum())
        iterations += 1
        is_penalization = not (masked[i, j] < -1e-9)

        if is_penalization:
            penalizations += 1
            edges = T.tour_edges(t)
            edge_arr = np.array(sorted(edges))
            lengths = D[edge_arr[:, 0], edge_arr[:, 1]]
            pens = penalty[edge_arr[:, 0], edge_arr[:, 1]]
            utility = lengths / (1.0 + pens)
            u, v = edge_arr[int(np.argmax(utility))]
            penalty[u, v] += 1
            penalty[v, u] += 1
            if debug_trace:
                debug.append((int(u), int(v), True))
        else:
            removed_a, removed_b = (t[i], t[i + 1]), (t[j], t[(j + 1) % n])
            length += float(delta[i, j])
            t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            if debug_trace:
                debug.append((i, j, False))
            if length < best_length - 1e-9:
                best_length, best_tour = length, t.copy()

        if keep_snapshots:
            snapshots.append(t.copy())
        if iterations % trace_every == 0 or evaluations >= budget:
            tr_it.append(evaluations)
            tr_len.append(length)
            tr_best.append(best_length)

    final_length = T.tour_length(t, D)                         # Rundungsfehler der Delta-Summen beseitigen
    best_length = T.tour_length(best_tour, D)
    return Run(best_tour, best_length, t, final_length, evaluations, iterations, penalizations, snapshots,
               np.array(tr_it), np.array(tr_len), np.array(tr_best), lam, debug)
