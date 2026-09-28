"""Szenario (wortgleich aus der Hill-Climbing-Demo, eingefrorene Werte) und Auswertung (Kennzahlen, Urteil, Tabu Search als
Vergleichspartner, Hill Climbing mit Neustarts, Sweeps, Streuung)."""

from dataclasses import replace

import numpy as np
import pytest

import gls_constants as C
import gls_evaluation as ev
import gls_scenario as S
import gls_tour as T

# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_instance_is_deterministic_seed_dependent_and_matches_the_frozen_hill_climbing_instance():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)
    inst, D = ev.instance(60, 0, 100000)
    assert inst.xy[0].tolist() == [50.0, 50.0] and float(inst.xy[1:].sum()) == pytest.approx(float(S.generate(60, 0, 100000).xy[1:].sum()))
    assert ev.reference_bound(60, 0, 100000) == pytest.approx(618.76, abs=0.05)


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(budget=200000))
    run = a.run
    assert a.bound < run.best_length <= run.final_length + 1e-9
    assert a.gap == pytest.approx(100 * (run.best_length - a.bound) / a.bound) and a.final_gap >= a.gap - 1e-9
    assert a.hc_gap > a.gap - 5.0                                    # ein Abstieg ist meist schlechter oder vergleichbar, nicht extrem besser
    assert run.evaluations >= a.settings.budget
    assert 0.0 <= a.penalization_rate <= 1.0
    assert a.tabu_gap > 0.0


def test_analysis_is_deterministic_given_the_chain_seed_and_the_chain_seed_changes_it():
    s = ev.Settings(n=30, budget=100000)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert np.array_equal(a.run.best_tour, b.run.best_tour) and a.gap == b.gap
    assert a.gap != c.gap or not np.array_equal(a.run.final_tour, c.run.final_tour)


def test_nearest_neighbor_start_is_deterministic_and_random_start_follows_the_chain_seed():
    s = ev.Settings(n=25, budget=20000, start="nearest")
    assert np.array_equal(ev.analyse(s).start_tour, ev.analyse(replace(s, chain_seed=3)).start_tour)
    r = ev.Settings(n=25, budget=20000)
    assert not np.array_equal(ev.analyse(r).start_tour, ev.analyse(replace(r, chain_seed=1)).start_tour)


def test_without_baselines_the_comparison_fields_are_empty_and_fast():
    a = ev.analyse(ev.Settings(n=20, budget=20000), with_baselines=False)
    assert a.hc is None and a.tabu is None and a.hcr_tour is None and a.hcr_starts == 0


def test_hill_climbing_with_restarts_uses_at_least_one_full_descent_and_stays_near_the_budget():
    inst, D = ev.instance(40, 0, 100000)
    single = T.descend(D, T.random_tour(len(D), np.random.default_rng(0)), "2opt", "first", keep_steps=False)
    tour, starts, used = ev.hill_climbing_restarts(D, 1000, 0)
    assert starts == 1 and used > 1000 and T.is_local_optimum(D, tour, "2opt")
    tour, starts, used = ev.hill_climbing_restarts(D, 5 * single.evaluations, 0)
    assert starts >= 3 and used <= 5 * single.evaluations + 2 * len(D) ** 2
    best_single = ev.hill_climbing_restarts(D, 1, 0)[0]
    assert T.tour_length(tour, D) <= T.tour_length(best_single, D) + 1e-9


# --- Urteil -------------------------------------------------------------------------------------------------------------------------------------


def _fake(gap, tabu_gap):
    class F:
        pass
    f = F()
    f.gap, f.tabu_gap = gap, tabu_gap
    return f


def test_verdict_codes():
    assert ev.verdict(_fake(1.0, 1.0 + ev.WIN_MARGIN + 0.1)) == "beats_tabu"
    assert ev.verdict(_fake(1.0 + ev.LOSE_MARGIN + 0.1, 1.0)) == "tabu_wins"
    assert ev.verdict(_fake(1.0, 1.2)) == "comparable"


def test_verdict_of_a_real_run_at_a_generous_budget():
    assert ev.verdict(ev.analyse(ev.Settings(budget=1000000))) in ("beats_tabu", "comparable", "tabu_wins")


# --- Sweeps und Tabellen ------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=20, budget=50000))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_min"] <= r["gap"] <= r["gap_max"] and r["gap_sd"] >= 0 and r["seconds"] > 0


def test_run_config_ignores_the_seeds_of_the_base_settings():
    a = ev.run_config(ev.Settings(n=15, budget=30000, seed=1, chain_seed=5))
    b = ev.run_config(ev.Settings(n=15, budget=30000, seed=999, chain_seed=0))
    assert all(a[k] == b[k] for k in a if not k.endswith("seconds"))


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("budget", ev.Settings(n=20), (100000, 1000000))
    assert rows[1]["gap"] <= rows[0]["gap"] + 1.0


def test_lambda_zero_is_the_no_penalty_ablation_and_matches_a_single_descent():
    row = ev.run_config(ev.Settings(n=20, budget=50000), lam=0.0)
    assert row["gap"] == pytest.approx(row["hc"], abs=1.0)            # eingefroren nach dem ersten lokalen Optimum, wie ein Abstieg


def test_scaling_table_structure(monkeypatch):
    monkeypatch.setattr(ev, "SCALING_N", (10, 20))
    tab = ev.scaling_table(ev.Settings(budget=30000))
    assert len(tab) == 2 and all([r["value"] for r in blk["rows"]] == [10, 20] for blk in tab) and tab[0]["label"] != tab[1]["label"]


def test_chain_spread_returns_one_value_per_chain_and_is_deterministic():
    a = ev.chain_spread(ev.Settings(n=15, budget=30000), 5)
    b = ev.chain_spread(ev.Settings(n=15, budget=30000), 5)
    assert len(a["gls"]) == len(a["tabu"]) == len(a["hc"]) == 5 and np.array_equal(a["gls"], b["gls"]) and np.array_equal(a["tabu"], b["tabu"])
