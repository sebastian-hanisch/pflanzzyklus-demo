"""Tests aus den Funden des Mutationslaufs (siehe README, Abschnitt Tests): jede Aussage schließt eine Lücke, die ein eingebauter Fehler offenließ."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate, stats

import pfz_constants as C
import pfz_orchard as O
import pfz_rules as R
import pfz_season as S
from pfz_exact import PlanInstance, solve_exact
from pfz_market import market_paths

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


def _loads(plan):
    return lambda t, stand, pool, m: np.full_like(stand, plan.get(t, 0.0))


# --- Saison ---------------------------------------------------------------------------------------------------------------------------

def test_the_last_day_counts():
    """Die Schleife über die Tage ließ den letzten Tag (Mi 23.12., Markttag) aus."""
    sp = S.SeasonParams(stock=10)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), S.markt_none, fixed_draws=_draws({26: 4}))
    assert profit[0] == pytest.approx(4 * 42.5 + 0.4 * 45 * 6) and info["sold_h"][0] == 4
    _, info2 = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _load_on(26, 5), fixed_draws=_draws({}, {26: 5}))
    assert info2["sold_m"][0] == 5 and info2["attended"][0] == 1


def test_a_load_of_one_tree_still_pays_the_market_day():
    """-140 Standgebühr - 2,5 - 0,8 + 45 + Rest 9 * 0,4 * 45 = 63,7."""
    sp = S.SeasonParams(stock=10)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _load_on(1, 1), fixed_draws=_draws({}, {1: 1}))
    assert profit[0] == pytest.approx(63.7) and info["attended"][0] == 1


def test_returns_are_taken_from_the_pool_on_a_later_market_day():
    """Tag 2: 5 laden, nichts verkauft (Pool 5). Tag 6 (Mi 02.12.): 3 aus dem Pool mitnehmen, 2 verkauft, 1 geht zurück; Tag 7: Hof verkauft 3.
    -156,5 - (140 + 2,4) + 90 + 135 + Rest 5 * 18 = 16,1."""
    sp = S.SeasonParams(stock=10, decay=0.0)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _loads({1: 5.0, 5: 3.0}), fixed_draws=_draws({6: 3}, {5: 2}))
    assert profit[0] == pytest.approx(16.1) and info["attended"][0] == 2 and info["sold_m"][0] == 2 and info["sold_h"][0] == 3


def test_returns_that_stay_in_the_pool_are_sold_later_without_cutting_cost():
    """Wie oben, Hof verkauft aber erst 1, dann 5 Bäume: Tag 7 nimmt zuerst die 2 Rückläufer aus dem Pool (keine Schlagkosten), dann 3 aus dem Stand.
    -156,5 - 142,4 + 90 + 45 + 90 + 3 * 42,5 + Rest 2 * 18 = 89,6. Fehlerhafter Pool-Bestand (Mitnahme addiert statt abgezogen) ergäbe 151,1."""
    sp = S.SeasonParams(stock=10, decay=0.0)
    profit, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _loads({1: 5.0, 5: 3.0}), fixed_draws=_draws({6: 1, 7: 5}, {5: 2}))
    assert profit[0] == pytest.approx(89.6) and info["left"][0] == 2 and info["sold_h"][0] == 6


def test_quality_loss_is_booked_on_market_sales_too():
    """Wie oben mit 10 % Wertverlust: die Rückläufer sind am Tag 6 vier Tage alt (Faktor 0,8): Verlust 2 * 45 * 0,2 = 18 auf dem Markt."""
    sp = S.SeasonParams(stock=10, decay=0.1)
    _, info = S.simulate_season(sp, np.ones((1, T)), 1, S.price_fixed(1.0), _loads({1: 5.0, 5: 3.0}), fixed_draws=_draws({}, {5: 2}))
    assert info["qloss"][0] == pytest.approx(18.0)


def test_hindsight_cannot_sell_more_trees_than_the_stock():
    sp = S.SeasonParams()
    value = S.hindsight(sp, np.ones((1, T)), 1.0, _draws({0: 3000}, {1: 500}))
    assert value[0] == pytest.approx(2000 * 42.5)                       # Hof zuerst (Marge 42,5 > 41,7), für den Markt bleibt nichts
    sp2 = S.SeasonParams(p_markt=50.0)                                  # Markt zuerst (46,7 > 42,5)
    assert S.hindsight(sp2, np.ones((1, T)), 1.0, _draws({0: 3000}, {1: 500}))[0] == pytest.approx(500 * 46.7 + 1500 * 42.5 - 140)


def test_hindsight_includes_the_last_market_day_and_does_not_attend_for_one_tree():
    sp = S.SeasonParams()
    assert S.hindsight(sp, np.ones((1, T)), 1.0, _draws({}, {26: 30}))[0] == pytest.approx(30 * 41.7 + 0.4 * 45 * 1970 - 140)
    one = S.hindsight(sp, np.ones((1, T)), 1.0, _draws({}, {1: 1}))
    assert one[0] == pytest.approx(0.4 * 45 * 2000)                      # ein Baum deckt die Standgebühr nicht: kein Markttag


def test_demand_factors_have_mean_one():
    """Die Korrektur -sigma^2/2 sichert den Mittelwert 1 (sonst verschiebt sich jede Nachfrage)."""
    u = S.draw_factors(S.SeasonParams(), 20000, 3)
    assert u.mean() == pytest.approx(1.0, abs=0.004)
    assert u.mean(axis=1).std() == pytest.approx(0.166, abs=0.006)        # Saisonfaktor (0,15) und Wetter (0,35 / Wurzel 27) zusammen


def _exact_gain_curve(sp, value, t, stand=2000):
    """Unabhängig: erwarteter Zusatznutzen des l-ten Baums mit der exakten Poisson-logNormal-Mischung (Quadratur)."""
    _, lam_m = S.base_rates(sp)
    dv = value[t + 1][stand] - value[t + 1][stand - 1]
    f_ret = (1 - sp.decay) if sp.back else 0.5

    def sf(l):
        g = lambda u: stats.poisson.sf(l - 1, lam_m[t] * u) * stats.lognorm.pdf(u, s=sp.sig_w, scale=np.exp(-sp.sig_w**2 / 2))
        return integrate.quad(g, 1e-9, 40, limit=200)[0]

    return np.array([sf(l) * (sp.p_markt - f_ret * dv) - ((1 - f_ret) * dv + sp.c_carry) for l in range(1, 201)])


@pytest.mark.parametrize("back,decay", [(True, 0.06), (True, 0.2), (False, 0.15)])
def test_newsvendor_load_matches_the_exact_mixture(back, decay):
    """Orakel: Menge und Standgebühr-Schwelle der Schattenpreisregel gegen die exakte Mischung (die Regel nutzt eine Negativbinomial-Näherung)."""
    sp = S.SeasonParams(back=back, decay=decay)
    value, _ = S.dp_prices(sp, grid=(1.0,))
    gains = _exact_gain_curve(sp, value, 1)
    total_gain = gains[gains > 0].sum()
    stand, pool, m = np.array([2000.0]), np.zeros((1, S.A_MAX)), np.array([1.0])
    load = S.markt_newsvendor(sp, value)(1, stand, pool, m)[0]
    assert abs(load - (gains > 0).sum()) <= 3
    above = S.markt_newsvendor(S.SeasonParams(back=back, decay=decay, fee=1.05 * total_gain), value)(1, stand, pool, m)[0]
    below = S.markt_newsvendor(S.SeasonParams(back=back, decay=decay, fee=0.9 * total_gain), value)(1, stand, pool, m)[0]
    assert above == 0.0 and below > 0


# --- Markt, LP, Hof, Regeln -----------------------------------------------------------------------------------------------------------

def test_the_last_market_year_follows_the_recurrence_too():
    """Die Schleife über die Jahre ließ das letzte Jahr aus (Marktpflanzung blieb auf dem Anfangswert 1)."""
    _, q = market_paths(40, 3, 1.3)
    assert np.any(q[:, -1] != 1.0) and np.all((q[:, -1] >= C.Q_CAP[0]) & (q[:, -1] <= C.Q_CAP[1]))


def test_the_lp_respects_the_demand_cap_in_every_year_including_the_last(mini_farm):
    """Die Nachfragegrenze des letzten Verkaufsjahres fehlte im LP."""
    farm = replace(mini_farm, window=3, demand=4.0)
    rng = np.random.default_rng(8)
    for _ in range(15):
        inst = PlanInstance(farm, rng.uniform(0.5, 2.0, 6), rng.uniform(5, 30, 2), 2, pi=rng.dirichlet(np.ones(3)), yld=1.0, pmax=20.0)
        sol = solve_exact(inst)
        assert len(sol.sold) == inst.J and sol.sold.max() <= farm.demand + 1e-7


def test_demand_shocks_have_mean_one(default_farm):
    """Die Korrektur -sigma^2/2 sichert den Mittelwert 1 der Jahresnachfrage."""
    p, q = market_paths(3000, 5, 1.1)
    sh = O.draw_shocks(default_farm, p, q, 6)
    assert sh.eps_d.mean() == pytest.approx(1.0, abs=0.004)
    assert sh.eps_d.std() == pytest.approx(np.sqrt(np.exp(0.04) - 1), abs=0.01)


def test_lueck_follows_the_hand_formula_for_a_single_weak_cohort():
    """Nur der Jahrgang y-1 hat Bäume (X). Gesucht ist P mit 3 * Absatz = Ankünfte aus y-1 (Jahre y+8, y+9, y+10) + künftige Jahrgänge
    mit der Sollmenge + P * yld * (PI[8] + PI[9] + PI[10]); die Formel unabhängig mit der Rohtabelle PI gerechnet."""
    farm = O.make_farm()
    ny = C.PRE + C.H
    shocks = O.Shocks(1, np.full((1, ny), 0.8), np.zeros((1, ny), dtype=int), np.ones((1, C.H)), np.ones((1, ny)), np.ones((1, ny)), 1)
    pstar, yld = O.planting_target(farm), O.expected_yield(farm)
    y, x = 25, 1000.0
    n_hist = np.zeros((1, ny))
    n_hist[:, y - 1] = x
    ctx = O.Context(farm, shocks, n_hist, pstar, yld, 1.0)
    have = 0.0
    for j in (y + 8, y + 9, y + 10):
        have += x * O.PI[j - (y - 1)]
        for i in range(y + 1, j - 5):                                    # künftige Jahrgänge, die im Jahr j ankommen können (j - i >= 6)
            have += pstar * yld * O.PI[j - i]
    weight = yld * (O.PI[8] + O.PI[9] + O.PI[10])
    assert R.rule_lueck(y, ctx, 1.0)[0] == pytest.approx((3 * 2000 - have) / weight, rel=1e-9)
