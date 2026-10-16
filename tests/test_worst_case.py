"""Worst-Case am Slider-Maximum (DEMO-PLAYBOOK §1c): fängt Hänger und unerwartetes Aufblähen. Keine Zeit-Assertion, nur ein großzügiges Netz."""

import time

import numpy as np

import pfz_constants as C
import pfz_evaluation as ev

NET_SECONDS = 120.0


def test_analysis_at_all_maxima_finishes_and_is_finite():
    t0 = time.time()
    a = ev.analyse(ev.OrchardSettings(C.PHI_MAX, C.GAMMA_MAX, C.DRY_MAX, C.WINDOW_MAX, C.DEMAND_MAX))
    assert time.time() - t0 < NET_SECONDS
    assert a.problems == [] and all(np.isfinite(s.profit) for s in a.stats.values())


def test_exact_reference_at_all_maxima_finishes():
    t0 = time.time()
    ex = ev.exact_comparison(ev.OrchardSettings(C.PHI_MAX, C.GAMMA_MAX, C.DRY_MAX, C.WINDOW_MAX, C.DEMAND_MAX), n_paths=3)
    assert time.time() - t0 < NET_SECONDS
    assert np.isfinite(ex.optimum_vs_gleich) and all(np.isfinite(v) for v in ex.gap_mean.values())


def test_season_check_at_all_maxima_finishes():
    t0 = time.time()
    sa = ev.analyse_season(ev.SeasonSettings(share=C.SHARE_MAX, fee=C.FEE_MAX, decay=C.DECAY_MAX, rho=C.RHO_MAX, kappa=C.KAPPA_MAX, back="nein"))
    assert time.time() - t0 < NET_SECONDS
    assert all(np.isfinite([l.value, l.se, l.pct]).all() for l in sa.levers)
