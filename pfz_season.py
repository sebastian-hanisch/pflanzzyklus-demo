"""Akt 2: die Saison. 2 000 Bäume, 27 Tage (27.11. bis 23.12.2026), Hofverkauf täglich, Wochenmarkt mittwochs und samstags.

Der Hof verkauft stehende Bäume auf Bestellung (schlagen bei Verkauf). Für den Markt werden Bäume geschlagen und mitgeführt; Rückläufer
verlieren an Wert und gehen, wenn der Hof sie zurücknimmt, zuerst in den Hofverkauf. Einheiten wie in pfz_orchard: kleine Funktionen
mit expliziten Ein- und Ausgaben, `simulate_season` ruft sie nur der Reihe nach auf.
"""

import datetime
from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, poisson

import pfz_constants as C
from pfz_orchard import sell_oldest

T = C.SEASON_DAYS
_D0 = datetime.date(*C.SEASON_START)
DATES = [_D0 + datetime.timedelta(i) for i in range(T)]
WEEKDAY = np.array([d.weekday() for d in DATES])   # Montag = 0
A_MAX = 30                                          # Altersklassen des geschlagenen Bestands (Tage)
_ADVENT = ((datetime.date(2026, 11, 29), 0.6), (datetime.date(2026, 12, 6), 0.8), (datetime.date(2026, 12, 13), 1.0),
           (datetime.date(2026, 12, 20), 1.3))
_WD_HOF = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.6, 5: 1.6, 6: 1.5}
_WD_MARKT = {2: 0.6, 5: 1.6}


def _advent_factor(d: datetime.date) -> float:
    for limit, f in _ADVENT:
        if d < limit:
            return f
    return 1.5


ADV = np.array([_advent_factor(d) for d in DATES])
W_HOF = np.array([_WD_HOF[w] for w in WEEKDAY]) * ADV
W_MARKT = np.array([_WD_MARKT.get(w, 0.0) for w in WEEKDAY]) * ADV
IS_MARKT = W_MARKT > 0
N_MARKT_DAYS = int(IS_MARKT.sum())


@dataclass(frozen=True)
class SeasonParams:
    rho: float = C.DEFAULT_RHO          # Gesamtnachfrage beim Richtpreis / Bestand
    kappa: float = C.DEFAULT_KAPPA      # Preisreaktion: Rate = Rate0 * max(0, 1 - kappa * (m - 1))
    share: float = C.DEFAULT_SHARE / 100.0
    fee: float = float(C.DEFAULT_FEE)
    decay: float = C.DEFAULT_DECAY / 100.0
    back: bool = True                   # Hof nimmt geschlagene Rückläufer zurück und verkauft sie
    stock: int = C.S0
    p_hof: float = C.SEASON_DEFAULTS["p_hof"]
    p_markt: float = C.SEASON_DEFAULTS["p_markt"]
    c_cut: float = C.SEASON_DEFAULTS["c_cut"]
    c_carry: float = C.SEASON_DEFAULTS["c_carry"]
    v_carry: float = C.SEASON_DEFAULTS["v_carry"]
    sig_w: float = C.SEASON_DEFAULTS["sig_w"]
    sig_z: float = C.SEASON_DEFAULTS["sig_z"]


def base_rates(sp: SeasonParams):
    """Erwartete Kundenzahl je Tag bei Preisfaktor 1: (Hof, Markt)."""
    demand = sp.rho * sp.stock
    lam_m = demand * sp.share * W_MARKT / W_MARKT.sum()
    lam_h = demand * (1 - sp.share) * W_HOF / W_HOF.sum()
    return lam_h, lam_m


def price_effect(m, kappa):
    """Mengenfaktor der Nachfrage bei Preisfaktor m."""
    return np.maximum(0.0, 1.0 - kappa * (np.asarray(m) - 1.0))


def quality(ages: np.ndarray, decay: float) -> np.ndarray:
    """Wertfaktor eines geschlagenen Baums nach `ages` Tagen: 2 Tage ohne Verlust, danach `decay` je Tag."""
    return np.maximum(0.0, 1.0 - decay * np.maximum(0, ages - 2))


def draw_factors(sp: SeasonParams, n: int, seed: int) -> np.ndarray:
    """Nachfragefaktor U [n, T] = Tagesfaktor (Wetter) mal Saisonfaktor, Mittel 1."""
    rng = np.random.default_rng(seed)
    w = np.exp(sp.sig_w * rng.standard_normal((n, T)) - sp.sig_w**2 / 2)
    z = np.exp(sp.sig_z * rng.standard_normal((n, 1)) - sp.sig_z**2 / 2)
    return w * z


def draw_demand(sp: SeasonParams, u: np.ndarray, seed: int, m: float):
    """Poisson-Nachfrage (Hof, Markt) je Tag bei festem Preisfaktor m; gemeinsame Ziehung für Regeln und Hindsight."""
    rng = np.random.default_rng(seed)
    lam_h, lam_m = base_rates(sp)
    pe = price_effect(m, sp.kappa)
    return rng.poisson(lam_h[None, :] * u * pe), rng.poisson(lam_m[None, :] * u * pe)


def simulate_season(sp: SeasonParams, u: np.ndarray, seed: int, price_fn, markt_fn, fixed_draws=None):
    """Eine Saison je Zeile von `u`. Gibt (Gewinn [n], Kennzahlen) zurück. Gewinn inklusive Restbäumen (v_carry)."""
    n = u.shape[0]
    rng = np.random.default_rng(seed)
    lam_h, lam_m = base_rates(sp)
    stand = np.full(n, float(sp.stock))
    pool = np.zeros((n, A_MAX))
    f = quality(np.arange(A_MAX), sp.decay)
    profit, attended, sold_h, sold_m, qloss, spoiled = (np.zeros(n) for _ in range(6))
    for t in range(T):
        total = stand + pool.sum(axis=1)
        m = price_fn(t, total)
        pe = price_effect(m, sp.kappa)
        if IS_MARKT[t]:
            load = np.minimum(np.floor(markt_fn(t, stand, pool, m)), total)
            go = load > 0
            attended += go
            profit -= sp.fee * go
            pool_after, taken = sell_oldest(pool, load)            # zuerst die ältesten Rückläufer mitnehmen
            taken_by_age = pool - pool_after
            fresh = np.minimum(load - taken, stand)
            carried = taken_by_age.copy()
            carried[:, 0] += fresh
            stand = stand - fresh
            profit -= sp.c_cut * fresh + sp.c_carry * (taken + fresh)
            demand_m = fixed_draws[1][:, t] if fixed_draws is not None else rng.poisson(lam_m[t] * u[:, t] * pe)
            unsold, sold = sell_oldest(carried, demand_m)
            sold_by_age = carried - unsold
            profit += sp.p_markt * m * (sold_by_age * f).sum(axis=1)
            qloss += sp.p_markt * m * (sold_by_age * (1 - f)).sum(axis=1)
            sold_m += sold
            pool = pool_after + unsold
        demand_h = fixed_draws[0][:, t] if fixed_draws is not None else rng.poisson(lam_h[t] * u[:, t] * pe)
        if sp.back:
            rest_pool, sold_pool = sell_oldest(pool, demand_h)
            by_age = pool - rest_pool
            profit += sp.p_hof * m * (by_age * f).sum(axis=1)
            qloss += sp.p_hof * m * (by_age * (1 - f)).sum(axis=1)
            pool = rest_pool
        else:
            sold_pool = np.zeros(n)
        rest = demand_h - sold_pool
        from_stand = np.minimum(rest, stand)
        profit += (sp.p_hof * m - sp.c_cut) * from_stand
        sold_h += sold_pool + from_stand
        stand = stand - from_stand
        spoiled += pool[:, -1]
        pool = np.concatenate([np.zeros((n, 1)), pool[:, :-1]], axis=1)
    profit += sp.v_carry * sp.p_hof * stand
    return profit, dict(attended=attended, sold_h=sold_h, sold_m=sold_m, left=stand, qloss=qloss, spoiled=spoiled)


# --- Preisregeln --------------------------------------------------------------------------------------------------------------------------

def price_fixed(m: float):
    """Fester Preisfaktor über die ganze Saison."""
    return lambda t, total: np.full_like(total, m, dtype=float)


def _lognormal_nodes(sigma: float, nodes: int = 11):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return np.exp(sigma * x - sigma**2 / 2), w / w.sum()


def dp_prices(sp: SeasonParams, grid=C.SEASON_PRICE_GRID, dmax: int = 400):
    """Rückwärtsrechnung über (Tag, Restbestand), nur Hofverkauf. Gibt (Wert V [T+1, S+1], Preisfaktor [T, S+1]) zurück.

    Die Nachfrage jedes Tages ist eine Poisson-Mischung mit logNormal-Faktor; Persistenz des Saisonfaktors wird ignoriert, es wird nicht gelernt.
    """
    lam_h, _ = base_rates(sp)
    u, wq = _lognormal_nodes(np.sqrt(sp.sig_w**2 + sp.sig_z**2))
    s = np.arange(sp.stock + 1)
    d = np.arange(dmax + 1)
    mind = np.minimum(d[None, :], s[:, None])
    value = np.zeros((T + 1, sp.stock + 1))
    value[T] = sp.v_carry * sp.p_hof * s
    policy = np.ones((T, sp.stock + 1))
    for t in range(T - 1, -1, -1):
        nxt = value[t + 1][s[:, None] - mind]
        best = np.full(sp.stock + 1, -np.inf)
        for m in grid:
            lam = lam_h[t] * float(price_effect(m, sp.kappa)) * u
            pm = (poisson.pmf(d[None, :], lam[:, None]) * wq[:, None]).sum(axis=0)
            pm = pm / pm.sum()
            val = ((sp.p_hof * m - sp.c_cut) * mind + nxt) @ pm
            better = val > best
            best = np.where(better, val, best)
            policy[t] = np.where(better, m, policy[t])
        value[t] = best
    return value, policy


def price_dp(policy: np.ndarray):
    """Preisfaktor nach (Tag, Restbestand) aus der Rückwärtsrechnung."""
    smax = policy.shape[1] - 1
    return lambda t, total: policy[t][np.clip(total.astype(int), 0, smax)]


# --- Marktregeln --------------------------------------------------------------------------------------------------------------------------

def markt_none(t, stand, pool, m):
    """Kein Markt."""
    return np.zeros_like(stand)


def markt_fixed(load: float):
    """Jeden Markttag dieselbe Menge mitnehmen."""
    return lambda t, stand, pool, m: np.full_like(stand, float(load))


def markt_newsvendor(sp: SeasonParams, value: np.ndarray):
    """Menge je Markttag nach dem Schattenpreis eines Baums (aus der Hof-Rückwärtsrechnung): lädt, solange der erwartete Zusatznutzen
    des letzten Baums positiv ist; geht nur, wenn die Summe die Standgebühr übersteigt."""
    cv2 = np.exp(sp.sig_w**2) - 1.0
    k = 1.0 / cv2
    f_ret = (1 - sp.decay) if sp.back else 0.5
    smax = value.shape[1] - 1
    ls = np.arange(1, 501)
    _, lam_m = base_rates(sp)

    def fn(t, stand, pool, m):
        total = np.clip((stand + pool.sum(axis=1)).astype(int), 1, smax)
        dv = value[t + 1][total] - value[t + 1][total - 1]
        mu = lam_m[t] * price_effect(m, sp.kappa)
        q = nbinom.sf(ls[None, :] - 1, k, (k / (k + mu))[:, None])          # P(Nachfrage >= l)
        pm = sp.p_markt * m
        gain = q * (pm - f_ret * dv)[:, None] - ((1 - f_ret) * dv + sp.c_carry)[:, None]
        pos = gain > 0
        load = pos.sum(axis=1).astype(float)
        load = np.where(np.where(pos, gain, 0.0).sum(axis=1) > sp.fee, load, 0.0)
        return np.minimum(load, stand + pool.sum(axis=1))

    return fn


# --- Schranke bei Vorabkenntnis --------------------------------------------------------------------------------------------------------

def hindsight(sp: SeasonParams, u: np.ndarray, m: float, draws):
    """Gewinn bei Vorabkenntnis der Nachfrage (Preisfaktor m fest): Menge frei verteilbar, Markttage per Aufzählung gewählt, kein Verderb."""
    dh, dm = draws
    n = u.shape[0]
    days = np.where(IS_MARKT)[0]
    best = np.full(n, -np.inf)
    marg_m = sp.p_markt * m - sp.c_cut - sp.c_carry
    marg_h = sp.p_hof * m - sp.c_cut
    for mask in range(1 << len(days)):
        sel = [days[i] for i in range(len(days)) if mask >> i & 1]
        d_m = dm[:, sel].sum(axis=1) if sel else np.zeros(n)
        d_h = dh.sum(axis=1)
        if marg_m >= marg_h:
            sm = np.minimum(d_m, sp.stock)
            sh = np.minimum(d_h, sp.stock - sm)
        else:
            sh = np.minimum(d_h, sp.stock)
            sm = np.minimum(d_m, sp.stock - sh)
        left = sp.stock - sm - sh
        val = sm * marg_m + sh * marg_h + sp.v_carry * sp.p_hof * left - sp.fee * len(sel) * (sm > 0)
        best = np.maximum(best, val)
    return best
