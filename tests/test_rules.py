"""Pflanzregeln an handgesetzten Kontexten (eine Einheit je Regel) und der Zweig-Test: jede Regel weicht von jeder anderen ab."""

import numpy as np
import pytest

import pfz_constants as C
import pfz_orchard as O
import pfz_rules as R


def _ctx(n=2, p=1.0, q=1.0, phi=1.0):
    farm = O.make_farm()
    ny = C.PRE + C.H
    shocks = O.Shocks(n, np.full((n, ny), 0.8), np.zeros((n, ny), dtype=int), np.ones((n, C.H)), np.full((n, ny), p), np.full((n, ny), q), 1)
    pstar, yld = O.planting_target(farm), O.expected_yield(farm)
    return O.Context(farm, shocks, np.full((n, ny), pstar * yld), pstar, yld, phi), pstar


def test_gleich_plants_the_target_whatever_happens():
    ctx, pstar = _ctx(p=3.0)
    assert R.rule_gleich(20, ctx, 1.0).tolist() == [pstar, pstar]


def test_folgend_scales_with_the_price_power_gamma():
    ctx, pstar = _ctx(p=1.5)
    assert R.rule_folgend(20, ctx, 1.0) == pytest.approx([1.5 * pstar] * 2)
    assert R.rule_folgend(20, ctx, 0.5) == pytest.approx([1.5 ** 0.5 * pstar] * 2)


def test_anti_plants_less_when_the_price_is_high_and_is_clipped():
    ctx, pstar = _ctx(p=1.5)
    assert R.rule_anti(20, ctx, 1.0) == pytest.approx([pstar / 1.5] * 2)
    ctx2, _ = _ctx(p=2.0)
    assert R.rule_anti(20, ctx2, 2.0) == pytest.approx([C.RULE_CLIP[0] * pstar] * 2)     # 1/4 liegt unter der unteren Grenze 0,3


def test_folgend_is_clipped_at_twice_the_target():
    ctx, pstar = _ctx(p=5.0)
    assert R.rule_folgend(20, ctx, 1.0) == pytest.approx([C.RULE_CLIP[1] * pstar] * 2)


def test_voraus_uses_only_known_planting_years_and_the_market_response():
    # alle bekannten Marktpflanzungen (Jahre <= y) doppelt so groß, die übrigen (y+1, y+2) im Mittel 1:
    # Angebot in neun Jahren = 2 * (0,3 + 0,25 + 0,15 + 0,05) + (0,05 + 0,20) = 1,75; Preis = 1,75^(-phi)
    ctx, pstar = _ctx(q=2.0, phi=1.0)
    assert R.expected_supply_in(20, ctx) == pytest.approx([1.75, 1.75])
    assert R.rule_voraus(20, ctx, 1.0) == pytest.approx([pstar / 1.75] * 2)


def test_nach_replaces_half_of_the_target_when_half_of_last_year_was_lost():
    ctx, pstar = _ctx()
    ctx.N[:, 19] *= 0.5
    assert R.rule_nach(20, ctx, 1.0) == pytest.approx([1.5 * pstar] * 2)
    ctx.N[:, 19] *= 3.0                                         # Überschuss im Vorjahr: kein negativer Ersatz
    assert R.rule_nach(20, ctx, 1.0) == pytest.approx([pstar] * 2)


def test_lueck_is_the_fixed_point_in_a_steady_state():
    """Gleichmäßig gepflanzte Vorjahre decken den Absatz genau: der Lückenfüller pflanzt dann die Sollmenge."""
    ctx, pstar = _ctx()
    assert R.rule_lueck(20, ctx, 1.0) == pytest.approx([pstar] * 2, rel=1e-9)


def test_lueck_fills_a_hole():
    ctx, pstar = _ctx()
    ctx.N[:, 16:20] *= 0.5                                      # vier schwache Jahrgänge (y-4 bis y-1), die um das Jahr y+9 ankommen
    assert np.all(R.rule_lueck(20, ctx, 1.0) > 1.2 * pstar)


def test_perfekt_follows_the_true_future_price():
    ctx, pstar = _ctx(p=1.5)
    assert R.rule_perfekt(20, ctx, 1.0) == pytest.approx([1.5 * pstar] * 2)


def test_make_rule_binds_gamma():
    ctx, pstar = _ctx(p=2.0)
    assert R.make_rule("folgend", 0.5)(20, ctx) == pytest.approx([2.0 ** 0.5 * pstar] * 2)
    assert set(R.RULE_FUNCS) == set(C.RULES)


def test_every_rule_differs_from_every_other_on_the_default_scenario(default_analysis):
    """Zweig-Test (DEMO-PLAYBOOK §0): keine Regel läuft ins Leere oder gleicht einer anderen."""
    profits = {r: default_analysis.stats[r].profit for r in C.RULES}
    for i, a in enumerate(C.RULES):
        for b in C.RULES[i + 1:]:
            assert abs(profits[a] - profits[b]) > 1e-6, (a, b)


def test_rules_that_react_to_the_price_equal_gleich_in_a_flat_market():
    """Preis konstant 1: folgend, anti und perfekt pflanzen genau die Sollmenge."""
    farm = O.make_farm()
    ny = C.PRE + C.H
    sh = O.draw_shocks(farm, np.ones((3, ny)), np.ones((3, ny)), 4)
    base = O.simulate(farm, sh, R.make_rule("gleich", 1.0), 1.0).cash
    for r in ("folgend", "anti", "perfekt"):
        assert O.simulate(farm, sh, R.make_rule(r, 1.0), 1.0).cash == pytest.approx(base)
