"""Orakel der exakten Referenz: LP gegen Aufzählung, gegen eine unabhängige Vorwärtsrechnung und gegen die Simulation im Erwartungswert."""

import numpy as np
import pytest

import pfz_constants as C
import pfz_orchard as O
from pfz_exact import PlanInstance, evaluate_plan, solve_exact
from pfz_market import market_paths
from pfz_rules import make_rule


def test_evaluate_plan_matches_the_hand_calculation(mini_instance):
    """Jahr 0: 5 Bäume (Vorgeschichte) verkauft: 50 - Pflege 0,5 * (10 - 5) = 47,5; Jahr 1: 5 verkauft: 50; Summe 97,5.
    Mit 4 Bäumen im Jahr 0 gepflanzt: 41,5 + 69 + 20 = 130,5."""
    value, sold, waste = evaluate_plan(mini_instance, np.array([0.0, 0.0]))
    assert value == pytest.approx(97.5) and sold.tolist() == [5.0, 5.0, 0.0, 0.0, 0.0] and waste.sum() == 0
    assert evaluate_plan(mini_instance, np.array([4.0]))[0] == pytest.approx(130.5)


def test_lp_equals_the_best_grid_plan_on_the_mini_instance(mini_instance):
    """Bei gleichem Preis ist "älteste zuerst" optimal; das LP muss dann das Maximum der Aufzählung treffen (hier 375 bei P = (15, 20))."""
    grid = np.arange(0.0, 20.01, 0.5)
    best = max(evaluate_plan(mini_instance, np.array([a, b]))[0] for a in grid for b in grid)
    sol = solve_exact(mini_instance)
    assert sol.status == "OPTIMAL"
    assert sol.objective == pytest.approx(best) == pytest.approx(375.0)
    assert sol.planting == pytest.approx([15.0, 20.0])


def test_lp_with_window_one_equals_the_forward_evaluation_of_its_own_plan(mini_farm, mini_pi):
    """Mit Fenster 1 gibt es keine Wahl in der Verkaufsreihenfolge: LP-Zielwert und Vorwärtsrechnung des LP-Plans müssen übereinstimmen."""
    from dataclasses import replace
    farm = replace(mini_farm, window=1)
    rng = np.random.default_rng(3)
    for _ in range(25):
        price = rng.uniform(0.3, 2.0, 6)
        hist = rng.uniform(0, 15, 2)
        pi = rng.dirichlet(np.ones(3))
        inst = PlanInstance(farm, price, hist, 3, pi=pi, yld=float(rng.uniform(0.5, 1.0)), pmax=float(rng.uniform(5, 25)))
        sol = solve_exact(inst)
        assert sol.objective == pytest.approx(evaluate_plan(inst, sol.planting)[0], rel=1e-9, abs=1e-7)


def test_lp_is_an_upper_bound_for_every_plan_on_random_mini_instances(mini_farm):
    rng = np.random.default_rng(4)
    for _ in range(25):
        price = rng.uniform(0.3, 2.0, 7)
        inst = PlanInstance(mini_farm, price, rng.uniform(0, 15, 2), 3, pi=rng.dirichlet(np.ones(4)), yld=0.8, pmax=15.0)
        opt = solve_exact(inst).objective
        for _ in range(20):
            plan = rng.uniform(0, 15, 3)
            assert evaluate_plan(inst, plan)[0] <= opt + 1e-7
            assert solve_exact(inst, fixed_planting=plan).objective <= opt + 1e-7
            assert solve_exact(inst, fixed_planting=plan).objective >= evaluate_plan(inst, plan)[0] - 1e-7


def test_fixed_planting_is_respected(mini_instance):
    sol = solve_exact(mini_instance, fixed_planting=np.array([3.0, 7.0]))
    assert sol.planting == pytest.approx([3.0, 7.0])


def _deterministic(farm, shocks):
    surv = np.where(np.arange(shocks.surv.shape[1])[None, :] < C.PRE, shocks.surv, O.expected_survival(farm))
    return O.Shocks(shocks.n, surv, shocks.shift, np.ones_like(shocks.eps_d), shocks.p, shocks.q, shocks.seed)


@pytest.mark.parametrize("rule", ["gleich", "folgend", "lueck"])
def test_simulation_in_expected_mode_equals_the_independent_forward_evaluation(default_farm, rule):
    """Zwei getrennte Rechenwege für dasselbe Modell: Simulation (expected=True) und evaluate_plan über alle H Jahre."""
    p, q = market_paths(2, 100000, 1.2)
    sh = _deterministic(default_farm, O.draw_shocks(default_farm, p, q, 5))
    res = O.simulate(default_farm, sh, make_rule(rule, 1.0), 1.2, expected=True)
    for i in range(2):
        hist = sh.surv[i, : C.PRE] * O.planting_target(default_farm) * default_farm.q_a
        inst = PlanInstance(default_farm, sh.p[i, C.PRE:], hist, C.H)
        value, sold, waste = evaluate_plan(inst, res.planted[i, C.PRE:])
        assert value == pytest.approx(res.cash[i].sum(), rel=1e-9)
        assert sold == pytest.approx(res.sold[i]) and waste == pytest.approx(res.waste[i])


def test_the_lp_optimum_is_never_below_any_rule_on_a_market_path(default_farm):
    p, q = market_paths(1, 100000, 1.2)
    sh = _deterministic(default_farm, O.draw_shocks(default_farm, p, q, 5))
    hl = C.EXACT_HORIZON
    inst = PlanInstance(default_farm, sh.p[0, C.PRE: C.PRE + hl + len(O.PI) + default_farm.window],
                        sh.surv[0, : C.PRE] * O.planting_target(default_farm) * default_farm.q_a, hl)
    opt = solve_exact(inst)
    assert opt.status == "OPTIMAL"
    for rule in ("gleich", "folgend", "voraus", "nach", "lueck", "perfekt"):
        plan = O.simulate(default_farm, sh, make_rule(rule, 1.0), 1.2, expected=True).planted[0, C.PRE: C.PRE + hl]
        assert solve_exact(inst, fixed_planting=plan).objective <= opt.objective + 1e-6
