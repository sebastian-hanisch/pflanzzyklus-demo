"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Die Zufallsströme von numpy sind zwischen Versionen nicht garantiert gleich; die Prüfungen haben deshalb Bänder um die README-Zahl
(mehrere Standardfehler), keine exakten Werte. Strukturelle Aussagen (Nullen, Vorzeichen, Reihenfolgen) sind exakt."""

import numpy as np
import pytest

import pfz_constants as C
import pfz_evaluation as ev
import pfz_orchard as O
import pfz_season as S
from test_oracle_market import _max_root
from scipy.optimize import brentq

SWEEP_COL = {phi: i for i, phi in enumerate(C.PHI_SWEEP)}


@pytest.fixture(scope="module")
def sweep1():
    return ev.sweep_phi(ev.OrchardSettings())


@pytest.fixture(scope="module")
def sweep05():
    return ev.sweep_phi(ev.OrchardSettings(gamma=0.5))


def _d(rows, phi, rule):
    return rows[SWEEP_COL[phi]]["delta"][rule]


def test_following_costs_little_in_a_weak_market_and_a_lot_in_a_strong_one(sweep1):
    assert [round(sweep1[i]["spread"], 2) for i in (0, 1, 2, 3, 4, 5, 6)] == pytest.approx([0.12, 0.14, 0.20, 0.31, 0.59, 0.89, 1.08], abs=0.02)
    for phi, claim in ((0.5, -0.2), (0.8, -0.5), (1.0, -1.6)):
        assert _d(sweep1, phi, "folgend") == pytest.approx(claim, abs=0.5)
    for phi, claim, tol in ((1.1, -6.4, 1.5), (1.2, -27.5, 4.0), (1.3, -50.5, 4.0), (1.4, -61.7, 4.0)):
        assert _d(sweep1, phi, "folgend") == pytest.approx(claim, abs=tol)
    assert sweep1[4]["delta"]["folgend"] < -25


def test_cautious_following_costs_about_half_as_much(sweep05):
    for phi, claim, tol in ((1.1, -1.6, 1.0), (1.2, -11.2, 3.0), (1.3, -28.0, 4.0)):
        assert _d(sweep05, phi, "folgend") == pytest.approx(claim, abs=tol)


def test_foresight_from_public_numbers_never_beats_gleich(sweep1):
    values = [r["delta"]["voraus"] for r in sweep1]
    assert max(values) <= 0.3 and min(values) == pytest.approx(-4.8, abs=1.5)
    assert _d(sweep1, 1.1, "voraus") == pytest.approx(-2.7, abs=1.0)


def test_antizyclical_planting_is_always_worse(sweep1):
    values = [r["delta"]["anti"] for r in sweep1]
    assert max(values) < 0 and values[0] == pytest.approx(-1.2, abs=0.5) and values[-1] == pytest.approx(-8.6, abs=1.5)


def test_replanting_the_loss_costs_money_and_vanishes_without_droughts(sweep1):
    values = [r["delta"]["nach"] for r in sweep1]
    assert min(values) == pytest.approx(-4.1, abs=0.5) and max(values) == pytest.approx(-3.5, abs=0.5) and max(values) < -3.0
    assert ev.analyse(ev.OrchardSettings(dry=35)).stats["nach"].delta_pct == pytest.approx(-6.8, abs=1.0)
    assert ev.analyse(ev.OrchardSettings(dry=0)).stats["nach"].delta == pytest.approx(0.0, abs=1e-6)


def test_the_gap_filler_is_the_only_rule_that_beats_gleich_and_only_by_a_little(sweep1):
    values = [r["delta"]["lueck"] for r in sweep1]
    assert min(values) == pytest.approx(0.7, abs=0.3) and max(values) == pytest.approx(0.8, abs=0.3) and min(values) > 0.2
    a35 = ev.analyse(ev.OrchardSettings(dry=35))
    assert a35.stats["lueck"].delta_pct == pytest.approx(1.6, abs=0.5)
    assert ev.analyse(ev.OrchardSettings()).stats["lueck"].delta_se_pct < 0.1


def test_the_reference_rule_with_known_price_gains_only_in_a_strong_market(sweep1):
    assert _d(sweep1, 1.1, "perfekt") == pytest.approx(-2.4, abs=1.0)
    assert _d(sweep1, 1.3, "perfekt") == pytest.approx(1.3, abs=1.5) and _d(sweep1, 1.4, "perfekt") == pytest.approx(4.5, abs=2.0)
    assert _d(sweep1, 1.1, "perfekt") < 0 < _d(sweep1, 1.4, "perfekt")


def test_exact_reference_values():
    ex = ev.exact_comparison(ev.OrchardSettings())
    assert ex.n_paths == 10 and ex.optimum_vs_gleich == pytest.approx(3.8, abs=0.8) and ex.optimum_vs_gleich_se == pytest.approx(0.5, abs=0.3)
    assert ex.gap_mean["gleich"] == pytest.approx(3.7, abs=0.8) and ex.gap_mean["folgend"] == pytest.approx(8.0, abs=1.5)
    assert ex.gap_mean["lueck"] == pytest.approx(3.6, abs=0.8) and ex.gap_mean["lueck"] < ex.gap_mean["gleich"]
    weak = ev.exact_comparison(ev.OrchardSettings(phi=0.8))
    strong = ev.exact_comparison(ev.OrchardSettings(phi=1.3))
    assert weak.optimum_vs_gleich == pytest.approx(4.5, abs=1.0) and strong.optimum_vs_gleich == pytest.approx(6.2, abs=1.5)
    assert strong.gap_mean["folgend"] == pytest.approx(34.0, abs=5.0)


def test_sales_window_acts_like_a_store():
    w1 = ev.analyse(ev.OrchardSettings(window=1))
    w5 = ev.analyse(ev.OrchardSettings(window=5))
    assert w1.stats["folgend"].delta_pct == pytest.approx(-14.0, abs=2.0) and w1.stats["gleich"].lost == pytest.approx(216, abs=40)
    assert w5.stats["folgend"].delta_pct == pytest.approx(0.5, abs=1.0)


def test_the_standard_case_of_gleichmaessig(default_analysis):
    s = default_analysis.stats["gleich"]
    assert s.lost == pytest.approx(31, abs=8) and s.lost / 2000 * 100 == pytest.approx(1.6, abs=0.4) and s.waste == pytest.approx(38, abs=8)
    assert O.planting_target(O.make_farm()) == pytest.approx(3022, abs=1)
    assert default_analysis.spread == pytest.approx(0.31, abs=0.03)


def test_the_stability_threshold_of_the_market_is_1_09():
    assert brentq(lambda phi: _max_root(phi) - 1.0, 0.5, 2.0) == pytest.approx(1.09, abs=0.005)


@pytest.fixture(scope="module")
def season_default():
    return ev.analyse_season(ev.SeasonSettings())


def _lv(sa, key):
    return next(l for l in sa.levers if l.key == key)


def test_season_default_levers(season_default):
    sa = season_default
    assert sa.base == pytest.approx(68121, abs=600)
    assert _lv(sa, "markt").value == pytest.approx(10502, abs=700) and _lv(sa, "markt").pct == pytest.approx(15.4, abs=1.0)
    assert _lv(sa, "regel").value == pytest.approx(730, abs=250) and _lv(sa, "regel").pct == pytest.approx(1.1, abs=0.4)
    assert _lv(sa, "preis").value == pytest.approx(330, abs=200) and _lv(sa, "preis").pct == pytest.approx(0.5, abs=0.3)
    assert _lv(sa, "preisplan").value == 0.0 and _lv(sa, "preisplan").se == 0.0
    assert _lv(sa, "unkenntnis").value == pytest.approx(963, abs=250) and _lv(sa, "unkenntnis").pct == pytest.approx(1.4, abs=0.4)
    assert sa.details["qloss_nv"] == pytest.approx(15, abs=15) and sa.details["price_dp_left"] == pytest.approx(787, abs=60)
    assert all(abs(l.pct) < 3 for l in sa.levers if l.key != "markt")


def test_season_without_return_is_more_demanding():
    sa = ev.analyse_season(ev.SeasonSettings(back="nein", decay=15))
    assert sa.details["qloss_nv"] == pytest.approx(2202, abs=500)
    assert _lv(sa, "unkenntnis").value == pytest.approx(3407, abs=600) and _lv(sa, "unkenntnis").pct == pytest.approx(5.0, abs=1.0)


def test_season_with_scarce_stock_and_without_market():
    scarce = ev.analyse_season(ev.SeasonSettings(rho=1.3, kappa=2.5))
    assert _lv(scarce, "preisplan").value == pytest.approx(593, abs=250) and _lv(scarce, "preisplan").pct == pytest.approx(0.8, abs=0.4)
    none = ev.analyse_season(ev.SeasonSettings(share=0))
    assert _lv(none, "markt").value == 0 and _lv(none, "preis").value == pytest.approx(2955, abs=700) and _lv(none, "preis").pct == pytest.approx(3.6, abs=0.8)


def test_the_hindsight_stock_and_market_calendar_facts():
    assert S.N_MARKT_DAYS == 8 and S.T == 27 and C.S0 == 2000
