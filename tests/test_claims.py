"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds)
belegt. Positive UND negative Aussagen: bei knappem Budget ist Guided Local Search nahezu identisch mit Tabu Search - das steht hier ebenso
als Test wie die Stellen, an denen es klar gewinnt. Rechenzeiten sind nur als Größenordnung geprüft."""

from functools import lru_cache

import pytest

import gls_constants as C
import gls_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap"], 3.39, 1.5)
    near(std["tabu"], 4.19, 1.5)
    near(std["hc"], 7.85, 1.5)
    near(std["hcr"], 4.88, 1.5)


def test_gls_beats_tabu_search_at_the_standard_budget():
    std = cfg()
    assert std["gap"] < std["tabu"] - 0.3


# --- Budget-Sweep: nahezu identisch unten, klar davonziehend oben ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget,gap,tol", [
    (10000, 287.75, 100.0), (25000, 155.09, 60.0), (50000, 54.73, 25.0), (100000, 8.53, 4.0),
    (200000, 3.39, 1.5), (500000, 1.56, 1.2), (1000000, 0.86, 1.0), (2000000, 0.52, 1.0),
])
def test_budget_sweep_numbers(budget, gap, tol):
    near(cfg(budget=budget)["gap"], gap, tol)


def test_small_budgets_are_catastrophically_worse_than_a_single_descent():
    for budget in (10000, 25000, 50000):
        row = cfg(budget=budget)
        assert row["gap"] > row["hc"] + 20.0                       # weit schlechter als ein einzelner Hill-Climbing-Abstieg


def test_below_one_hundred_thousand_gls_is_nearly_identical_to_tabu_search():
    """Beide sind volle-Nachbarschaft-Verfahren - bei so wenigen Iterationen hat die Strafe noch keine Gelegenheit,
    sich von Tabus verfallender Sperre abzusetzen."""
    for budget in (10000, 25000, 50000):
        row = cfg(budget=budget)
        near(row["gap"], row["tabu"], max(5.0, 0.05 * row["tabu"]))


def test_from_two_hundred_thousand_gls_pulls_ahead_of_tabu_search_and_the_lead_grows():
    mid = cfg(budget=200000)
    large = cfg(budget=2000000)
    mid_edge = mid["tabu"] - mid["gap"]
    large_edge = large["tabu"] - large["gap"]
    assert mid_edge > 0.3 and large_edge > mid_edge - 0.3           # der Vorsprung schrumpft NICHT (anders als Tabus eigener Vorsprung vor HC-Neustarts)


# --- Lambda-Sweep -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lam,gap,tol", [(0.1, 7.74, 2.5), (1.0, 5.83, 2.0), (2.0, 4.65, 2.0), (5.0, 3.39, 1.5), (15.0, 4.28, 2.0), (30.0, 5.69, 2.0)])
def test_lambda_sweep_numbers_at_two_hundred_thousand(lam, gap, tol):
    near(cfg(lam=lam)["gap"], gap, tol)


def test_lambda_five_beats_both_too_weak_and_too_strong_a_penalty():
    weak = cfg(lam=0.1)["gap"]
    calibrated = cfg(lam=5.0)["gap"]
    strong = cfg(lam=30.0)["gap"]
    assert calibrated < weak and calibrated < strong


def test_a_very_weak_penalty_matches_a_single_hill_climbing_descent():
    near(cfg(lam=0.05)["gap"], cfg(lam=0.05)["hc"], 2.0)


# --- Große Instanz ----------------------------------------------------------------------------------------------------------------------------


def test_large_instance_shares_the_same_shared_budget_floor_as_tabu_search():
    row = ev.run_config(ev.Settings(n=200), budget=1000000)
    near(row["gap"], 306.74, 100.0)
    near(row["gap"], row["tabu"], max(10.0, 0.05 * row["tabu"]))    # kein Unterschied im Mechanismus, beide unterversorgt


def test_medium_instance_gls_clearly_beats_tabu_search():
    row = ev.run_config(ev.Settings(n=60), budget=5000 * 60)
    assert row["gap"] < row["tabu"] - 0.5


# --- Startlösung ------------------------------------------------------------------------------------------------------------------------------


def test_a_good_start_solution_clearly_helps():
    random_ = cfg(start="random")
    nearest = cfg(start="nearest")
    assert nearest["gap"] < random_["gap"] - 1.0


def test_bound_matches_the_frozen_reference():
    near(ev.reference_bound(60, 0, 100000), 618.76, 0.1)
