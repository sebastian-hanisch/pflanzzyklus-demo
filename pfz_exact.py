"""Exakte Referenz im Erwartungswert: Pflanzplan als lineares Programm (HiGHS), dazu eine unabhängige Vorwärtsrechnung desselben Modells.

Modell (deterministisch): Ausbeute im Erwartungswert, Nachfrage fest, Preisindex bekannt. Entscheidungen: Pflanzmenge je Jahr und
Verkaufsmenge je Jahr und Alter (zuerst die ältesten ist nur eine Regel, das LP wählt frei). Zielgröße: Summe der Jahresüberschüsse.
Das LP ist eine Obergrenze nur für diese Erwartungswert-Rechnung, nicht für den zufälligen Betrieb.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import pfz_constants as C
from pfz_orchard import PI, Farm, expected_yield, max_planting


@dataclass(frozen=True)
class PlanInstance:
    farm: Farm
    price: np.ndarray       # [J] Preisindex der Verkaufsjahre (Jahr 0 = erstes Jahr nach der Vorgeschichte)
    n_hist: np.ndarray      # [PRE] Verkaufsbäume der Pflanzjahrgänge der Vorgeschichte
    hl: int                 # Zahl der Pflanzjahre, über die entschieden wird (ab Jahr 0)
    pi: np.ndarray = field(default_factory=lambda: PI)
    yld: float = 0.0        # Verkaufsware je gepflanztem Baum (0: aus dem Hof berechnen)
    pmax: float = 0.0       # Obergrenze Pflanzmenge je Jahr (0: aus dem Hof berechnen)

    @property
    def J(self):
        return len(self.price)

    @property
    def y_eff(self):
        return self.yld or expected_yield(self.farm)

    @property
    def p_eff(self):
        return self.pmax or max_planting(self.farm)


def history_state(inst: PlanInstance):
    """Ankünfte der Vorgeschichte je Verkaufsjahr, Bestand zu Beginn von Jahr 0 und Zahl der schon angekommenen Bäume."""
    pre, w = len(inst.n_hist), inst.farm.window
    n_years = pre + inst.J
    arr = np.zeros(n_years)
    for h, n in enumerate(inst.n_hist):
        for a, pa in enumerate(inst.pi):
            if h + a < n_years:
                arr[h + a] += n * pa
    stock = np.zeros(w)
    for y in range(pre):
        stock[0] += arr[y]
        stock = np.concatenate([[0.0], stock[:-1]])
    return arr[pre:], stock, arr[:pre].sum()


def evaluate_plan(inst: PlanInstance, planting: np.ndarray, sales_rule: str = "oldest"):
    """Vorwärtsrechnung des Plans (zuerst die ältesten Bäume verkaufen). Gibt (Summe der Überschüsse, Verkäufe je Jahr, verworfen je Jahr)."""
    farm, J, w = inst.farm, inst.J, inst.farm.window
    pre_arr, stock, arrived_before = history_state(inst)
    yld = inst.y_eff
    plant = np.zeros(J)
    plant[: len(planting)] = planting
    arrivals = pre_arr.copy()
    for i, pm in enumerate(plant):
        for a, pa in enumerate(inst.pi):
            if i + a < J:
                arrivals[i + a] += pm * yld * pa
    total_n = inst.n_hist.sum() + yld * np.cumsum(plant)
    arrived = arrived_before + np.cumsum(arrivals)
    sold, waste = np.zeros(J), np.zeros(J)
    cash = 0.0
    for j in range(J):
        stock[0] += arrivals[j]
        s = 0.0
        left = farm.demand
        for k in range(w - 1, -1, -1):
            take = min(stock[k], left)
            stock[k] -= take
            left -= take
            s += take
        sold[j] = s
        care = farm.c_care * (max(total_n[j] - arrived[j], 0.0) + stock.sum())
        cash += s * farm.price * inst.price[j] - farm.c_plant * plant[j] - care
        waste[j] = stock[-1]
        stock = np.concatenate([[0.0], stock[:-1]])
    return cash, sold, waste


@dataclass(frozen=True)
class ExactSolution:
    status: str             # "OPTIMAL" oder Meldung
    objective: float | None
    planting: np.ndarray | None
    sold: np.ndarray | None


def solve_exact(inst: PlanInstance, fixed_planting: np.ndarray | None = None) -> ExactSolution:
    """Löst den Plan als LP mit HiGHS. Mit `fixed_planting` ist die Pflanzmenge vorgegeben, das LP wählt dann nur die Verkäufe."""
    farm, J, w, hl = inst.farm, inst.J, inst.farm.window, inst.hl
    yld, pi = inst.y_eff, inst.pi
    pre_arr, stock0, arrived_before = history_state(inst)
    n_var = hl + 2 * J * w
    ix = lambda j, k: hl + j * w + k            # Bestand x[j, k]
    isl = lambda j, k: hl + J * w + j * w + k   # Verkauf s[j, k]
    c = np.zeros(n_var)
    cum_pi = np.cumsum(pi)
    for i in range(hl):
        # Pflanzkosten + Pflege der noch nicht angekommenen Bäume (Anteil 1 - kumulierte Ankunft) in den Jahren ab i
        unarrived = sum(1.0 - (cum_pi[j - i] if j - i < len(cum_pi) else 1.0) for j in range(i, J))
        c[i] = farm.c_plant + farm.c_care * yld * unarrived
    for j in range(J):
        for k in range(w):
            c[isl(j, k)] = -farm.price * inst.price[j] - farm.c_care
            c[ix(j, k)] = farm.c_care
    n_eq = J + (w - 1) + (J - 1) * (w - 1)
    a_eq, b_eq = lil_matrix((n_eq, n_var)), np.zeros(n_eq)
    r = 0
    for j in range(J):               # x[j,0] = Ankünfte
        a_eq[r, ix(j, 0)] = 1.0
        for i in range(min(hl, j + 1)):
            if j - i < len(pi):
                a_eq[r, i] = -yld * pi[j - i]
        b_eq[r] = pre_arr[j]
        r += 1
    for k in range(1, w):            # Anfangsbestand
        a_eq[r, ix(0, k)] = 1.0
        b_eq[r] = stock0[k]
        r += 1
    for j in range(1, J):            # Fortschreibung des Bestands
        for k in range(1, w):
            a_eq[r, ix(j, k)] = 1.0
            a_eq[r, ix(j - 1, k - 1)] = -1.0
            a_eq[r, isl(j - 1, k - 1)] = 1.0
            r += 1
    a_ub, b_ub = lil_matrix((J * w + J, n_var)), np.zeros(J * w + J)
    r = 0
    for j in range(J):
        for k in range(w):
            a_ub[r, isl(j, k)], a_ub[r, ix(j, k)] = 1.0, -1.0
            r += 1
    for j in range(J):
        for k in range(w):
            a_ub[r, isl(j, k)] = 1.0
        b_ub[r] = farm.demand
        r += 1
    if fixed_planting is None:
        bounds = [(0.0, inst.p_eff)] * hl
    else:
        bounds = [(float(v), float(v)) for v in np.asarray(fixed_planting, dtype=float)[:hl]]
    bounds = bounds + [(0.0, None)] * (2 * J * w)
    res = linprog(c, A_ub=a_ub.tocsr(), b_ub=b_ub, A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return ExactSolution(res.message, None, None, None)
    # konstante Anteile: Pflege der bereits gepflanzten, noch nicht angekommenen Vorgeschichte
    cum_pre = np.cumsum(pre_arr)
    const = -farm.c_care * sum(max(inst.n_hist.sum() - arrived_before - cum_pre[j], 0.0) for j in range(J))
    planting = res.x[:hl]
    sold = np.array([res.x[isl(j, 0): isl(j, 0) + w].sum() for j in range(J)])
    return ExactSolution("OPTIMAL", float(-res.fun + const), planting, sold)
