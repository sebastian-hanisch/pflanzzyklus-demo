"""Gemeinsame Fixtures: Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pfz_orchard as O  # noqa: E402
from pfz_exact import PlanInstance  # noqa: E402


@pytest.fixture
def mini_farm():
    """Hof ohne Zufall: Absatz 10, Verkaufsfenster 2, jeder gepflanzte Baum wird Verkaufsware, Preis 10, Pflanzung 1, Pflege 0,5 je Baum und Jahr."""
    return O.Farm(demand=10, window=2, p_dry=0.0, sig_d=0.0, s0=1.0, dry_lo=0.2, dry_hi=0.6, q_a=1.0, price=10.0, c_plant=1.0,
                  c_care=0.5, pmax_factor=2.0)


@pytest.fixture
def mini_pi():
    """Ankunft 1 oder 2 Jahre nach der Pflanzung, je zur Hälfte."""
    return np.array([0.0, 0.5, 0.5])


@pytest.fixture
def mini_instance(mini_farm, mini_pi):
    """Ein Vorgeschichts-Jahrgang mit 10 Bäumen, 5 Verkaufsjahre, Preisindex 1, zwei Pflanzjahre, Obergrenze 20."""
    return PlanInstance(mini_farm, np.ones(5), np.array([10.0]), 2, pi=mini_pi, yld=1.0, pmax=20.0)


@pytest.fixture(scope="session")
def default_farm():
    return O.make_farm()


@pytest.fixture(scope="session")
def default_analysis():
    import pfz_evaluation as ev
    return ev.analyse(ev.OrchardSettings())
