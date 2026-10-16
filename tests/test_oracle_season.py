"""Akt 2: Einheiten der Saison an Handrechnungen, Orakel (Hindsight, Rückwärtsrechnung) und Zweig-Tests."""

import numpy as np
import pytest
from scipy import integrate, stats

import pfz_constants as C
import pfz_season as S

T = S.T


def _draws(day_h=None, day_m=None):
    dh, dm = np.zeros((1, T)), np.zeros((1, T))
    for t, v in (day_h or {}).items():
        dh[0, t] = v
    for t, v in (day_m or {}).items():
        dm[0, t] = v
    return dh, dm


def _load_on(day, amount):
    return lambda t, stand, pool, m: np.full_like(stand, amount if t == day else 0.0)


def test_calendar_has_27_days_and_8_market_days():
    assert T == 27 and S.DATES[0].isoformat() == "2026-11-27" and S.DATES[-1].isoformat() == "2026-12-23"
    assert S.N_MARKT_DAYS == 8
    assert [d.strftime("%d.%m.") for d, m in zip(S.DATES, S.IS_MARKT) if m] == ["28.11.", "02.12.", "05.12.", "09.12.", "12.12.", "16.12.", "19.12.", "23.12."]


def test_base_rates_add_up_to_the_total_demand_and_markt_only_on_market_days():
    sp = S.SeasonParams(rho=1.3, share=0.25)
    lam_h, lam_m = S.base_rates(sp)
    assert lam_h.sum() == pytest.approx(1.3 * 2000 * 0.75) and lam_m.sum() == pytest.approx(1.3 * 2000 * 0.25)
    assert np.all(lam_m[~S.IS_MARKT] == 0)


def test_quality_has_no_loss_for_two_days_then_decays_linearly_to_zero():
    assert S.quality(np.arange(5), 0.1) == pytest.approx([1.0, 1.0, 1.0, 0.9, 0.8])
    assert S.quality(np.array([40]), 0.1)[0] == 0.0


def test_price_effect_is_linear_and_never_negative():
    assert S.price_effect(1.0, 1.5) == 1.0 and S.price_effect(1.1, 1.5) == pytest.approx(0.85) and S.price_effect(2.0, 1.5) == 0.0


def test_hof_only_season_by_hand():
    """10 Bäume, 3 am Hof verkauft: 3 * (45 - 2,5) + Restwert 7 * 0,4 * 45 = 127,5 + 126 = 253,5."""
    sp = S.SeasonParams(stock=10)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), S.markt_none, fixed_draws=_draws({0: 3}))
    assert profit[0] == pytest.approx(253.5) and info["left"][0] == 7 and info["sold_h"][0] == 3


def test_market_day_with_return_to_the_farm_by_hand():
    """Tag 2 (Sa 28.11.): 5 laden, 2 verkauft; die 3 Rückläufer verkauft der Hof am Sonntag ohne Wertverlust (Alter 1).
    -140 Standgebühr - 5 * 2,5 Schlagen - 5 * 0,8 Mitführen + 2 * 45 + 3 * 45 + Rest 5 * 0,4 * 45 = 158,5."""
    sp = S.SeasonParams(stock=10)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _load_on(1, 5), fixed_draws=_draws({2: 3}, {1: 2}))
    assert profit[0] == pytest.approx(158.5)
    assert info["sold_m"][0] == 2 and info["sold_h"][0] == 3 and info["attended"][0] == 1 and info["left"][0] == 5


def test_return_trees_lose_value_with_age_when_the_farm_takes_them_back():
    """Rückläufer am Tag 5 (Alter 3) mit 10 % je Tag nach 2 Tagen: 2 * 45 * 0,9 = 81; -156,5 + 81 + 90 = 14,5."""
    sp = S.SeasonParams(stock=10, decay=0.1)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _load_on(1, 5), fixed_draws=_draws({4: 2}))
    assert profit[0] == pytest.approx(14.5) and info["qloss"][0] == pytest.approx(9.0)


def test_without_return_the_farm_cannot_sell_the_cut_trees():
    """Zweig: back=False. Dieselbe Saison, der Hof verkauft die 2 Bäume aus dem Stand: -156,5 + 2 * 42,5 + 3 * 18 = -17,5."""
    sp = S.SeasonParams(stock=10, decay=0.1, back=False)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _load_on(1, 5), fixed_draws=_draws({4: 2}))
    assert profit[0] == pytest.approx(-17.5) and info["sold_h"][0] == 2


def test_the_price_factor_changes_the_revenue_by_hand():
    sp = S.SeasonParams(stock=10)
    profit, _ = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.2), S.markt_none, fixed_draws=_draws({0: 3}))
    assert profit[0] == pytest.approx(3 * (45 * 1.2 - 2.5) + 0.4 * 45 * 7)


def test_hindsight_by_hand():
    """Hof-Nachfrage 100, Markt nur am 28.11. mit 30: Markt lohnt (30 * 41,7 > 140), Hof zuerst weil Marge 42,5 > 41,7:
    30 * 41,7 + 100 * 42,5 + 0,4 * 45 * 1 870 - 140 = 39 021; ohne Markt 38 450."""
    sp = S.SeasonParams()
    value = S.hindsight(sp, np.ones((1, T)), 1.0, _draws({0: 100}, {1: 30}))
    assert value[0] == pytest.approx(39021.0)
    value0 = S.hindsight(S.SeasonParams(fee=1e6), np.ones((1, T)), 1.0, _draws({0: 100}, {1: 30}))
    assert value0[0] == pytest.approx(38450.0)


def test_backward_induction_on_the_last_day_matches_numerical_integration():
    """Orakel: letzter Tag, Bestand 6, drei Preise; der Wert aus der Rückwärtsrechnung gegen eine unabhängige Quadratur über den
    logNormal-Faktor und die exakte Poisson-Verteilung."""
    sp = S.SeasonParams(stock=6)
    grid = (0.9, 1.0, 1.1)
    value, policy = S.dp_prices(sp, grid=grid)
    lam_h, _ = S.base_rates(sp)
    sigma = np.sqrt(sp.sig_w**2 + sp.sig_z**2)

    def expected(s, m):
        def inner(u):
            lam = lam_h[T - 1] * float(S.price_effect(m, sp.kappa)) * u
            d = np.arange(0, 200)
            pm = stats.poisson.pmf(d, lam)
            sold = np.minimum(d, s)
            val = (pm * ((sp.p_hof * m - sp.c_cut) * sold + sp.v_carry * sp.p_hof * (s - sold))).sum()
            return val * stats.lognorm.pdf(u, s=sigma, scale=np.exp(-sigma**2 / 2))
        return integrate.quad(inner, 1e-9, 30, limit=200)[0]

    for s in range(1, 7):
        best = max(expected(s, m) for m in grid)
        assert value[T - 1, s] == pytest.approx(best, rel=2e-3)
    assert value[T - 1, 0] == 0.0 and value[T, 3] == pytest.approx(0.4 * 45 * 3)


def test_value_of_a_tree_is_nonnegative_and_increasing_in_the_stock():
    value, policy = S.dp_prices(S.SeasonParams(stock=40), grid=(0.9, 1.0, 1.1))
    assert np.all(np.diff(value, axis=1) >= -1e-9)
    assert set(np.unique(policy)) <= {0.9, 1.0, 1.1}


def test_the_price_plan_is_constant_with_ample_stock_and_varies_when_stock_is_scarce():
    """Zweig-Test: der Preisplan weicht nur bei knappem Bestand vom festen Preis ab (Nullspalte im Hebel-Check, siehe README)."""
    ample, scarce = S.SeasonParams(), S.SeasonParams(rho=1.3, kappa=2.5)
    _, pol_a = S.dp_prices(ample)
    _, pol_s = S.dp_prices(scarce)
    assert set(np.unique(pol_a[:, 1500:])) == {1.05}          # ab 1 500 Bäumen an jedem Tag derselbe Preisfaktor
    assert len(np.unique(pol_s[:, 1300:])) > 1


def test_newsvendor_rule_loads_nothing_when_the_fee_is_prohibitive_and_something_otherwise():
    sp = S.SeasonParams()
    value, _ = S.dp_prices(sp, grid=(1.0,))
    stand, pool, m = np.array([2000.0]), np.zeros((1, S.A_MAX)), np.array([1.0])
    load = S.markt_newsvendor(sp, value)(1, stand, pool, m)
    assert load[0] > 20
    none = S.markt_newsvendor(S.SeasonParams(fee=1e7), value)(1, stand, pool, m)
    assert none[0] == 0.0


def test_newsvendor_loads_less_when_the_returns_are_worthless():
    """Ohne Rücknahme (Rückläufer wertlos bis auf den nächsten Markt) ist der kritische Anteil höher, die Menge kleiner oder gleich."""
    value, _ = S.dp_prices(S.SeasonParams(), grid=(1.0,))
    stand, pool, m = np.array([2000.0]), np.zeros((1, S.A_MAX)), np.array([1.0])
    back = S.markt_newsvendor(S.SeasonParams(back=True), value)(1, stand, pool, m)[0]
    no_back = S.markt_newsvendor(S.SeasonParams(back=False, decay=0.3), value)(1, stand, pool, m)[0]
    assert no_back <= back


def test_fixed_load_is_clipped_to_the_stock():
    sp = S.SeasonParams(stock=10)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), S.markt_fixed(500), fixed_draws=_draws())
    assert np.isfinite(profit).all() and info["left"][0] >= 0


def test_stock_is_conserved_over_a_random_season():
    """Verkauft (Hof + Markt) + Rest + Verwurf + Rückläufer im Pool ≤ Anfangsbestand."""
    sp = S.SeasonParams()
    u = S.draw_factors(sp, 20, 1)
    value, _ = S.dp_prices(sp, grid=(1.0,))
    _, info = S.simulate_season(sp, u, 2, S.price_fixed(1.0), S.markt_newsvendor(sp, value))
    assert np.all(info["sold_h"] + info["sold_m"] + info["left"] + info["spoiled"] <= sp.stock + 1e-9)
    assert np.all(info["left"] >= 0)
