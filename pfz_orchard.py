"""Hof über die Jahre: Pflanzjahrgänge, Ausfall, Ankunft der Verkaufsgröße, Verkaufsfenster, Kassenfluss.

Jede Einheit ist eine kleine Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); `simulate` ruft sie nur der Reihe nach auf.
Ein Baum durchläuft: gepflanzt im Jahr i -> überlebt (Ausfall) und ist Verkaufsware -> erreicht nach tau Jahren die Verkaufsgröße ->
kann `window` Jahre lang verkauft werden -> danach zu hoch und verworfen.
"""

from dataclasses import dataclass

import numpy as np

import pfz_constants as C


@dataclass(frozen=True)
class Farm:
    demand: float
    window: int
    p_dry: float
    sig_d: float
    s0: float
    dry_lo: float
    dry_hi: float
    q_a: float
    price: float
    c_plant: float
    c_care: float
    pmax_factor: float


def make_farm(demand: float = C.DEFAULT_DEMAND, window: int = C.DEFAULT_WINDOW, dry_percent: float = C.DEFAULT_DRY) -> Farm:
    return Farm(demand=float(demand), window=int(window), p_dry=dry_percent / 100.0, **C.FARM_DEFAULTS)


def arrival_profile() -> np.ndarray:
    """PI[a]: Wahrscheinlichkeit, dass ein Baum a Jahre nach der Pflanzung erstmals Verkaufsgröße hat (Mischung über die Verschiebung)."""
    pi = np.zeros(max(C.TAU) + max(C.SHIFT) + 2)
    for sh, ps in zip(C.SHIFT, C.PSHIFT):
        for tau, pt in zip(C.TAU, C.PTAU):
            pi[tau + sh] += ps * pt
    return pi


PI = arrival_profile()


def expected_survival(farm: Farm) -> float:
    """Überleben eines Pflanzjahrgangs im Erwartungswert (normales Überleben, im Dürrejahr zusätzlicher Ausfall)."""
    return farm.s0 * (1 - farm.p_dry * (farm.dry_lo + farm.dry_hi) / 2)


def expected_yield(farm: Farm) -> float:
    """Verkaufsware je gepflanztem Baum im Erwartungswert (Überleben mit Dürrejahren mal Verkaufsware)."""
    return expected_survival(farm) * farm.q_a


def planting_target(farm: Farm) -> float:
    """Sollpflanzmenge je Jahr, die im Erwartungswert genau den Absatz deckt."""
    return farm.demand / expected_yield(farm)


def max_planting(farm: Farm) -> float:
    return farm.pmax_factor * planting_target(farm)


def expected_arrivals(trees, i: int, j: int):
    """Erwartete Ankünfte im Jahr j aus `trees` Verkaufsbäumen des Pflanzjahres i."""
    a = j - i
    if 0 <= a < len(PI):
        return trees * PI[a]
    return trees * 0.0


@dataclass
class Shocks:
    """Zufall eines Auswertungslaufs: gemeinsam für alle Regeln (gepaarter Vergleich)."""
    n: int
    surv: np.ndarray        # [n, PRE+H] Überleben des Pflanzjahres
    shift: np.ndarray       # [n, PRE+H] Verschiebung der Ankunft des Pflanzjahres
    eps_d: np.ndarray       # [n, H] Nachfragefaktor
    p: np.ndarray           # [n, PRE+H] Marktpreisindex
    q: np.ndarray           # [n, PRE+H] Marktpflanzung
    seed: int


def draw_shocks(farm: Farm, p: np.ndarray, q: np.ndarray, seed: int) -> Shocks:
    n, ny = p.shape
    rng = np.random.default_rng(seed)
    dry = rng.random((n, ny)) < farm.p_dry
    sev = rng.uniform(farm.dry_lo, farm.dry_hi, (n, ny))
    surv = farm.s0 * (1 - dry * sev)
    shift = rng.choice(np.array(C.SHIFT), size=(n, ny), p=np.array(C.PSHIFT))
    eps_d = np.exp(farm.sig_d * rng.standard_normal((n, C.H)) - farm.sig_d**2 / 2)
    return Shocks(n, surv, shift, eps_d, p, q, int(rng.integers(0, 2**31 - 1)))


def split_arrivals(rng, trees: np.ndarray) -> np.ndarray:
    """Verteilt Verkaufsbäume zufällig auf die Ankunftsalter TAU (Binomial-Aufteilung, Summe bleibt erhalten)."""
    rem = np.round(trees).astype(np.int64)
    out = np.zeros((len(trees), len(C.TAU)))
    left = 1.0
    for i, pt in enumerate(C.PTAU):
        k = rem if i == len(C.PTAU) - 1 else rng.binomial(rem, min(1.0, pt / left))
        out[:, i] = k
        rem = rem - k
        left -= pt
    return out


def place_arrivals(arr: np.ndarray, y: int, trees: np.ndarray, shift: np.ndarray, rng, expected: bool) -> None:
    """Trägt die Ankünfte eines Pflanzjahres in die Jahresspalte von `arr` ein."""
    if expected:
        for a, pa in enumerate(PI):
            if pa > 0:
                arr[:, y + a] += trees * pa
        return
    for s in C.SHIFT:
        idx = np.where(shift == s)[0]
        if len(idx) == 0:
            continue
        counts = split_arrivals(rng, trees[idx])
        for ti, tau in enumerate(C.TAU):
            arr[idx, y + tau + s] += counts[:, ti]


def sell_oldest(stock: np.ndarray, demand: np.ndarray):
    """Verkauft aus dem Wartebestand [n, window] (Spalte k = Jahre seit Erreichen der Größe) zuerst die ältesten Bäume."""
    rev = stock[:, ::-1]
    before = np.cumsum(rev, axis=1) - rev
    sold_rev = np.clip(demand[:, None] - before, 0, rev)
    return stock - sold_rev[:, ::-1], sold_rev[:, ::-1].sum(axis=1)


def age_stock(stock: np.ndarray):
    """Ein Jahr weiter: Bäume in der letzten Spalte werden zu hoch und verworfen. Gibt (neuer Bestand, verworfene Bäume) zurück."""
    waste = stock[:, -1].copy()
    new = np.concatenate([np.zeros((stock.shape[0], 1)), stock[:, :-1]], axis=1)
    return new, waste


@dataclass
class Context:
    farm: Farm
    shocks: Shocks
    N: np.ndarray           # [n, PRE+H] Verkaufsbäume des Pflanzjahres (bis y-1 gefüllt)
    pstar: float
    yld: float
    phi: float              # Preisflexibilität des Marktes (für die Regel "Vorausschau")

    @property
    def n(self):
        return self.shocks.n

    @property
    def p(self):
        return self.shocks.p

    @property
    def q(self):
        return self.shocks.q


@dataclass
class Result:
    cash: np.ndarray        # [n, H]
    sold: np.ndarray
    waste: np.ndarray
    lost: np.ndarray        # entgangene Verkäufe (Nachfrage minus Verkauf)
    planted: np.ndarray     # [n, PRE+H]
    arrivals: np.ndarray    # [n, PRE+H+...] Ankünfte der Verkaufsgröße je Jahr


def simulate(farm: Farm, shocks: Shocks, rule, phi: float, expected: bool = False) -> Result:
    """Ein Lauf über PRE + H Jahre. `rule(y, ctx)` liefert die Pflanzmenge des Jahres y (ab y = PRE).

    expected=True: Ankünfte im Erwartungswert (keine Binomial-Aufteilung); mit Nachfrage- und Ausfallfaktor 1 ist der Lauf dann deterministisch.
    """
    n, ny = shocks.n, C.PRE + C.H
    rng = np.random.default_rng(shocks.seed)
    pstar, yld, pmax = planting_target(farm), expected_yield(farm), max_planting(farm)
    ctx = Context(farm, shocks, np.zeros((n, ny)), pstar, yld, phi)
    planted = np.zeros((n, ny))
    arr = np.zeros((n, ny + max(C.TAU) + 3))
    stock = np.zeros((n, farm.window))
    cash, sold_y, waste_y, lost_y = (np.zeros((n, C.H)) for _ in range(4))
    for y in range(ny):
        p_y = np.full(n, pstar) if y < C.PRE else np.clip(rule(y, ctx), 0.0, pmax)
        planted[:, y] = p_y
        ctx.N[:, y] = p_y * shocks.surv[:, y] * farm.q_a
        place_arrivals(arr, y, ctx.N[:, y], shocks.shift[:, y], rng, expected)
        stock[:, 0] += arr[:, y]
        t = y - C.PRE
        if t >= 0:
            demand = farm.demand * shocks.eps_d[:, t]
            stock, sold = sell_oldest(stock, demand)
            revenue = sold * farm.price * shocks.p[:, y]
            unarrived = np.maximum(ctx.N[:, : y + 1].sum(axis=1) - arr[:, : y + 1].sum(axis=1), 0.0)
            care = farm.c_care * (unarrived + stock.sum(axis=1))
            cash[:, t] = revenue - farm.c_plant * p_y - care
            sold_y[:, t], lost_y[:, t] = sold, demand - sold
            stock, waste_y[:, t] = age_stock(stock)
        else:
            stock, _ = age_stock(stock)
    return Result(cash, sold_y, waste_y, lost_y, planted, arr)


def summarize(res: Result) -> dict:
    """Jahresmittel je Pfad über das Auswertungsfenster (ab BURN)."""
    sl = slice(C.BURN, C.H)
    return dict(
        profit=res.cash[:, sl].mean(axis=1),
        lost=res.lost[:, sl].mean(axis=1),
        waste=res.waste[:, sl].mean(axis=1),
        sold=res.sold[:, sl].mean(axis=1),
        planted=res.planted[:, C.PRE + C.BURN:].mean(axis=1),
    )


def check_result(farm: Farm, shocks: Shocks, res: Result) -> list:
    """Unabhängiger Prüfer: kein Baum wird öfter verkauft als angekommen ist, nie mehr als die Nachfrage, Pflanzmenge in den Grenzen."""
    problems = []
    tol = 1e-6
    demand = farm.demand * shocks.eps_d
    if np.any(res.sold > demand + tol):
        problems.append("Verkauf größer als die Nachfrage")
    if np.any(res.sold < -tol) or np.any(res.waste < -tol):
        problems.append("negative Verkaufs- oder Verwurfsmenge")
    arrived = np.cumsum(res.arrivals[:, C.PRE: C.PRE + C.H], axis=1) + res.arrivals[:, : C.PRE].sum(axis=1, keepdims=True)
    gone = np.cumsum(res.sold + res.waste, axis=1)
    if np.any(gone > arrived + 1e-6 * (1 + arrived)):
        problems.append("mehr Bäume verkauft oder verworfen, als angekommen sind")
    if np.any(res.planted < -tol) or np.any(res.planted > max_planting(farm) + tol):
        problems.append("Pflanzmenge außerhalb der Grenzen")
    return problems
