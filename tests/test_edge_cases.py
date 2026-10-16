"""Randfälle: Extremwerte aller Regler, Entartungen des Hofs, Pfadzahl 1, Verkaufsfenster 1, alle Preise gleich."""

import itertools

import numpy as np
import pytest

import pfz_constants as C
import pfz_evaluation as ev
import pfz_orchard as O
import pfz_season as S
from pfz_market import market_paths
from pfz_rules import make_rule

CORNERS = [
    dict(phi=C.PHI_MIN), dict(phi=C.PHI_MAX), dict(gamma=C.GAMMA_MIN), dict(gamma=C.GAMMA_MAX), dict(dry=C.DRY_MIN), dict(dry=C.DRY_MAX),
    dict(window=C.WINDOW_MIN), dict(window=C.WINDOW_MAX), dict(demand=C.DEMAND_MIN), dict(demand=C.DEMAND_MAX),
]


@pytest.mark.parametrize("override", CORNERS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_every_slider_corner_gives_finite_results_and_a_quiet_checker(override):
    a = ev.analyse(ev.OrchardSettings(**override))
    assert a.problems == []
    assert all(np.isfinite([s.profit, s.delta_pct, s.lost, s.waste, s.planted]).all() for s in a.stats.values())


def test_all_minima_and_all_maxima_together():
    lo = ev.OrchardSettings(C.PHI_MIN, C.GAMMA_MIN, C.DRY_MIN, C.WINDOW_MIN, C.DEMAND_MIN)
    hi = ev.OrchardSettings(C.PHI_MAX, C.GAMMA_MAX, C.DRY_MAX, C.WINDOW_MAX, C.DEMAND_MAX)
    for o in (lo, hi):
        a = ev.analyse(o)
        assert a.problems == [] and all(np.isfinite(s.profit) for s in a.stats.values())


def test_without_droughts_nachpflanzen_equals_gleich():
    """Dürre 0 %: kein Jahrgang bleibt unter dem Erwartungswert, der Ersatz ist null, beide Regeln pflanzen genau gleich."""
    a = ev.analyse(ev.OrchardSettings(dry=0))
    assert a.stats["nach"].delta == pytest.approx(0.0, abs=1e-6)


def test_window_one_forces_a_sale_in_the_year_of_arrival():
    farm = O.make_farm(window=1)
    p, q = market_paths(3, 1, 1.1)
    sh = O.draw_shocks(farm, p, q, 2)
    res = O.simulate(farm, sh, make_rule("gleich", 1.0), 1.1)
    assert O.check_result(farm, sh, res) == [] and res.waste.sum() > 0


def test_a_single_path_works():
    farm = O.make_farm()
    p, q = market_paths(1, 1, 1.2)
    sh = O.draw_shocks(farm, p, q, 2)
    for r in C.RULES:
        res = O.simulate(farm, sh, make_rule(r, 1.0), 1.2)
        assert np.isfinite(res.cash).all() and O.check_result(farm, sh, res) == []


def test_a_flat_price_makes_all_rules_equal_in_profit_terms():
    farm = O.make_farm()
    ny = C.PRE + C.H
    sh = O.draw_shocks(farm, np.ones((2, ny)), np.ones((2, ny)), 5)
    base = O.simulate(farm, sh, make_rule("gleich", 1.0), 1.0).cash
    for r in ("folgend", "anti", "perfekt"):
        assert O.simulate(farm, sh, make_rule(r, 2.0), 1.0).cash == pytest.approx(base)


def test_planting_never_exceeds_the_area_limit_even_for_unbounded_rules():
    farm = O.make_farm(dry_percent=C.DRY_MAX)
    p, q = market_paths(20, 3, 1.4)
    sh = O.draw_shocks(farm, p, q, 4)
    for r in ("nach", "lueck"):
        res = O.simulate(farm, sh, make_rule(r, 1.0), 1.4)
        assert res.planted.max() <= O.max_planting(farm) + 1e-6 and res.planted.min() >= 0


SEASON_CORNERS = [dict(share=0), dict(share=50), dict(fee=0), dict(fee=600), dict(decay=0), dict(decay=30), dict(rho=0.6), dict(rho=1.5),
                  dict(kappa=0.5), dict(kappa=3.0), dict(back="nein")]


@pytest.mark.parametrize("override", SEASON_CORNERS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_season_sliders_at_their_limits_keep_the_stock_balance(override):
    ss = ev.SeasonSettings(**override)
    sp = ss.params()
    u = S.draw_factors(sp, 40, 1)
    value, _ = S.dp_prices(sp, grid=(1.0,))
    for markt in (S.markt_none, S.markt_fixed(60), S.markt_newsvendor(sp, value)):
        profit, info = S.simulate_season(sp, u, 3, S.price_fixed(1.0), markt)
        assert np.isfinite(profit).all()
        assert np.all(info["sold_h"] + info["sold_m"] + info["left"] + info["spoiled"] <= sp.stock + 1e-9) and np.all(info["left"] >= 0)


def test_no_market_share_means_no_market_sales():
    sp = ev.SeasonSettings(share=0).params()
    u = S.draw_factors(sp, 20, 1)
    _, info = S.simulate_season(sp, u, 3, S.price_fixed(1.0), S.markt_fixed(100))
    assert info["sold_m"].sum() == 0


def test_all_corner_combinations_of_the_season_run_through_the_hindsight():
    for share, back in itertools.product((0, 50), ("ja", "nein")):
        sp = ev.SeasonSettings(share=share, back=back).params()
        u = S.draw_factors(sp, 10, 1)
        draws = S.draw_demand(sp, u, 3, 1.0)
        assert np.isfinite(S.hindsight(sp, u, 1.0, draws)).all()
