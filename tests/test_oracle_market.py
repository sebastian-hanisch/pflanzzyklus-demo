"""Marktmodell (Spinnweb): Form, Reproduzierbarkeit und Vergleich mit der linearen Theorie (unabhängiger Rechenweg über Polynomnullstellen)."""

import numpy as np
import pytest
from scipy.optimize import brentq

import pfz_constants as C
from pfz_market import market_paths, price_spread


def _max_root(phi: float) -> float:
    """Linearisierung ln p_t = -phi * sum_k w_k ln p_(t-k) (eta = 1): z^12 + phi * sum_k w_k z^(12-k) = 0; Betrag der größten Nullstelle."""
    coeff = np.zeros(max(C.TAU) + 1)
    coeff[0] = 1.0
    for tau, w in zip(C.TAU, C.PTAU):
        coeff[tau] += phi * w
    return float(np.abs(np.roots(coeff)).max())


def test_paths_have_the_right_shape_mean_one_and_are_positive():
    p, q = market_paths(30, 3, 1.1)
    assert p.shape == q.shape == (30, C.PRE + C.H)
    assert np.all(p > 0) and np.all(q >= C.Q_CAP[0]) and np.all(q <= C.Q_CAP[1])
    assert p.mean(axis=1) == pytest.approx(np.ones(30))


def test_paths_are_reproducible_and_depend_on_the_seed():
    a, _ = market_paths(5, 11, 1.2)
    b, _ = market_paths(5, 11, 1.2)
    c, _ = market_paths(5, 12, 1.2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_price_spread_grows_with_the_market_reaction():
    spreads = [price_spread(market_paths(60, 5, phi)[0]) for phi in C.PHI_SWEEP]
    assert all(x < y for x, y in zip(spreads, spreads[1:])), spreads


def test_linear_theory_puts_the_stability_threshold_at_1_09():
    """Oracle: größte Nullstelle der charakteristischen Gleichung; Schwelle bei Betrag 1."""
    assert _max_root(0.8) < 1.0 < _max_root(1.3)
    threshold = brentq(lambda phi: _max_root(phi) - 1.0, 0.5, 2.0)
    assert threshold == pytest.approx(1.09, abs=0.005)


def test_without_noise_the_simulation_follows_the_theory(monkeypatch):
    """Ohne Stöße klingt die Störung unter der Schwelle ab und schwingt darüber bis zur Begrenzung."""
    monkeypatch.setattr(C, "SIG_MARKET_DEMAND", 0.0)
    monkeypatch.setattr(C, "SIG_MARKET_SUPPLY", 0.0)
    assert price_spread(market_paths(10, 3, 0.8)[0]) < 0.005
    assert price_spread(market_paths(10, 3, 1.2)[0]) > 0.2
