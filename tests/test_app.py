"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knöpfe, Permalink-Grenzen, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gls_constants as C
import gls_evaluation as ev

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(gls_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if gls_step != 1:
        at.select_slider(key="gls_step").set_value(gls_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Beste Tour") == "3.3 %" and _metric(at, "Tabu Search, gleiches Budget") == "3.7 %" and _metric(at, "HC mit Neustarts (voller Rescan)") == "3.9 %"
    assert any("Gleichauf" in s.value for s in at.info) or any("Besser als" in s.value for s in at.success) or any("besser" in s.value for s in at.warning)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["lambda_slider"] == p["lam"] and at.session_state["budget_select"] == p["budget"] and at.session_state["n_slider"] == p["n"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("n", [10, 60])
def test_every_step_runs(step, n):
    at = _run(n_slider=n, budget_select=50000, gls_step=step)
    _ok(at)
    assert at.get("plotly_chart") and at.session_state["gls_step"] == step


def test_step_two_iteration_slider_and_play_search():
    at = _run(gls_step=2, budget_select=50000)
    _ok(at)
    lv = next(s for s in at.slider if s.key == "gls_iter")
    assert lv.value == lv.max
    lv.set_value(0).run()
    _ok(at)
    at2 = _run(gls_step=2, budget_select=50000)
    next(b for b in at2.button if b.label == "▶️ Suche abspielen").click().run()
    _ok(at2)


def test_play_runs_through_all_steps_without_duplicate_keys():
    at = _run(n_slider=20, budget_select=50000)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_lambda_extremes_run():
    for lam in (C.LAMBDA_MIN, C.LAMBDA_MAX):
        _ok(_run(lambda_slider=lam, budget_select=50000))


def test_dice_buttons_change_the_seeds():
    at = _run(budget_select=50000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_c = at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Kette würfeln").click().run()
    _ok(at)
    assert at.session_state["chain_seed_input"] != old_c


@pytest.mark.parametrize("kw", [dict(n_slider=200, budget_select=100000), dict(n_slider=10, ballung_slider=100, budget_select=20000),
                                 dict(lambda_slider=C.LAMBDA_MIN, budget_select=50000), dict(lambda_slider=C.LAMBDA_MAX, budget_select=50000),
                                 dict(start_radio="nearest", budget_select=50000)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["lam"] = "9999"
    at.query_params["budget"] = "12345"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50
    assert at.session_state["lambda_slider"] == C.LAMBDA_MAX and at.session_state["budget_select"] == C.DEFAULT_BUDGET


def test_sweeps_run_on_demand():
    at = _run(n_slider=10, budget_select=20000)
    at.selectbox(key="sweep_select").set_value("lam").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    monkeypatch.setitem(ev.SWEEP_VALUES, "budget", (10000, 20000))
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    monkeypatch.setattr(ev, "SCALING_POLICIES", (("Budget 4 Tausend", lambda n: 40000), ("Budget 300 · Stopps", lambda n: 3000 * n)))
    at = _run(n_slider=10, budget_select=20000)
    for key, flag in (("budget_start", "budget_on"), ("spread_start", "spread_on"), ("scaling_start", "scaling_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=20000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Das Budget reicht für genug teure Iterationen" in m.value for m in at.markdown)
