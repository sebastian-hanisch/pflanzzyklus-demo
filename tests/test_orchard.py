"""Einheiten des Hofs an Handrechnungen: Ankunftsprofil, Ertrag, Verkauf, Altern, Aufteilung der Ankünfte, Prüfer."""

import numpy as np
import pytest

import pfz_constants as C
import pfz_orchard as O


def test_arrival_profile_sums_to_one_and_has_the_hand_computed_mean():
    pi = O.arrival_profile()
    assert pi.sum() == pytest.approx(1.0)
    # Mittel der Ankunftsalter: 7*0,05 + 8*0,2 + 9*0,3 + 10*0,25 + 11*0,15 + 12*0,05 = 9,4; die Verschiebung ist symmetrisch
    assert (np.arange(len(pi)) * pi).sum() == pytest.approx(9.4)
    assert pi[:7].sum() == pytest.approx(0.25 * 0.05)   # vor Jahr 7 kommt nur tau = 7 mit Verschiebung -1 an (Jahr 6)


def test_expected_yield_and_planting_target_by_hand():
    farm = O.make_farm()
    # 0,88 * (1 - 0,15 * 0,4) * 0,8 = 0,66176
    assert O.expected_survival(farm) == pytest.approx(0.88 * 0.94)
    assert O.expected_yield(farm) == pytest.approx(0.66176)
    assert O.planting_target(farm) == pytest.approx(2000 / 0.66176)
    assert O.max_planting(farm) == pytest.approx(1.5 * O.planting_target(farm))


def test_sell_oldest_takes_the_oldest_trees_first():
    stock = np.array([[2.0, 3.0, 4.0]])           # Spalte k = Jahre seit Erreichen der Größe, k = 2 ist der älteste
    new, sold = O.sell_oldest(stock, np.array([5.0]))
    assert new.tolist() == [[2.0, 2.0, 0.0]] and sold.tolist() == [5.0]


def test_sell_oldest_is_limited_by_the_stock():
    new, sold = O.sell_oldest(np.array([[1.0, 1.0]]), np.array([10.0]))
    assert new.tolist() == [[0.0, 0.0]] and sold.tolist() == [2.0]


def test_age_stock_discards_the_last_column_and_shifts():
    new, waste = O.age_stock(np.array([[1.0, 2.0, 3.0]]))
    assert new.tolist() == [[0.0, 1.0, 2.0]] and waste.tolist() == [3.0]


def test_age_stock_with_window_one_discards_everything():
    new, waste = O.age_stock(np.array([[4.0]]))
    assert new.tolist() == [[0.0]] and waste.tolist() == [4.0]


def test_split_arrivals_conserves_the_trees_and_follows_the_profile():
    rng = np.random.default_rng(1)
    counts = O.split_arrivals(rng, np.full(2000, 1000.0))
    assert np.all(counts.sum(axis=1) == 1000)
    assert counts.mean(axis=0) / 1000 == pytest.approx(np.array(C.PTAU), abs=0.01)


def test_place_arrivals_expected_adds_the_profile_in_the_right_years():
    arr = np.zeros((1, 40))
    O.place_arrivals(arr, 3, np.array([100.0]), np.array([0]), np.random.default_rng(0), expected=True)
    assert arr[0, 3 + 9] == pytest.approx(100 * O.PI[9])
    assert arr.sum() == pytest.approx(100.0)


def test_place_arrivals_random_respects_the_shift():
    arr = np.zeros((2, 40))
    O.place_arrivals(arr, 0, np.array([50.0, 50.0]), np.array([-1, 1]), np.random.default_rng(0), expected=False)
    assert arr[0, :6].sum() == 0 and arr[0, 6:13].sum() == 50          # Verschiebung -1: Jahre 6 bis 11
    assert arr[1, :8].sum() == 0 and arr[1, 8:14].sum() == 50          # Verschiebung +1: Jahre 8 bis 13


def test_summarize_uses_only_the_evaluation_window():
    cash = np.zeros((1, C.H))
    cash[0, : C.BURN] = 1e9
    cash[0, C.BURN:] = 2.0
    res = O.Result(cash, np.zeros((1, C.H)), np.zeros((1, C.H)), np.zeros((1, C.H)), np.ones((1, C.PRE + C.H)), np.zeros((1, 10)))
    assert O.summarize(res)["profit"].tolist() == [2.0]


def test_the_checker_accepts_a_real_run_and_finds_planted_errors(default_farm):
    from pfz_market import market_paths
    from pfz_rules import make_rule
    p, q = market_paths(4, 5, 1.1)
    sh = O.draw_shocks(default_farm, p, q, 6)
    res = O.simulate(default_farm, sh, make_rule("gleich", 1.0), 1.1)
    assert O.check_result(default_farm, sh, res) == []
    res.sold[0, 5] += 1e5                                     # mehr verkauft als Nachfrage und Ankünfte
    problems = O.check_result(default_farm, sh, res)
    assert any("Nachfrage" in m for m in problems) and any("angekommen" in m for m in problems)
    res.planted[1, 20] = 1e9
    assert any("Grenzen" in m for m in O.check_result(default_farm, sh, res))
