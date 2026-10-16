"""Pflanzregeln: je Regel eine Funktion (y, ctx) -> Pflanzmenge des Jahres y für alle Pfade. Kein versteckter Zustand."""

import numpy as np

import pfz_constants as C
from pfz_orchard import PI, Context, expected_arrivals


def _clip(ctx: Context, amount):
    return np.clip(amount, C.RULE_CLIP[0] * ctx.pstar, C.RULE_CLIP[1] * ctx.pstar)


def rule_gleich(y: int, ctx: Context, gamma: float):
    """Jedes Jahr die Sollmenge."""
    return np.full(ctx.n, ctx.pstar)


def rule_folgend(y: int, ctx: Context, gamma: float):
    """Pflanzmenge proportional zum heutigen Preisindex hoch gamma."""
    return _clip(ctx, ctx.pstar * ctx.p[:, y] ** gamma)


def rule_anti(y: int, ctx: Context, gamma: float):
    """Pflanzmenge umgekehrt zum heutigen Preisindex: bei niedrigem Preis mehr."""
    return _clip(ctx, ctx.pstar * ctx.p[:, y] ** (-gamma))


def expected_supply_in(y: int, ctx: Context, horizon: int = 9):
    """Marktangebot im Jahr y + horizon, soweit die Pflanzjahre schon bekannt sind (spätere Jahre zählen mit dem Mittel 1)."""
    s = np.zeros(ctx.n)
    for tau, wt in zip(C.TAU, C.PTAU):
        i = y + horizon - tau
        s += wt * (ctx.q[:, i] if i <= y else 1.0)
    return s


def rule_voraus(y: int, ctx: Context, gamma: float):
    """Pflanzmenge nach dem aus den öffentlichen Pflanzzahlen berechneten Preis in neun Jahren."""
    price_then = expected_supply_in(y, ctx) ** (-ctx.phi)
    return _clip(ctx, ctx.pstar * price_then ** gamma)


def rule_nach(y: int, ctx: Context, gamma: float):
    """Sollmenge plus Ersatz für den Fehlbetrag des Vorjahrgangs."""
    short = np.maximum(0.0, ctx.pstar * ctx.yld - ctx.N[:, y - 1]) / ctx.yld
    return ctx.pstar + short


def rule_lueck(y: int, ctx: Context, gamma: float):
    """Pflanzt den Rest, der für die erwarteten Ankünfte der Jahre y+8 bis y+10 zum Absatz fehlt (künftige Jahre mit Sollmenge angesetzt)."""
    have = np.zeros(ctx.n)
    weight = 0.0
    for j in (y + 8, y + 9, y + 10):
        for i in range(max(0, y - 14), y):
            have += expected_arrivals(ctx.N[:, i], i, j)
        for i in range(y + 1, j - (min(C.TAU) + min(C.SHIFT)) + 1):   # künftige Jahrgänge, die im Jahr j schon ankommen können (früheste Ankunft: tau 7 mit Verschiebung -1)
            have += expected_arrivals(ctx.pstar * ctx.yld, i, j)
        weight += expected_arrivals(ctx.yld, y, j)
    return np.clip((3 * ctx.farm.demand * (1 + C.LUECKEN_BUFFER) - have) / max(weight, 1e-9), 0.0, 1e12)


def rule_perfekt(y: int, ctx: Context, gamma: float):
    """Referenz: kennt den wirklichen Preisindex der Ankunftsjahre (gewichtet nach PTAU) und pflanzt danach."""
    ny = ctx.p.shape[1]
    target = sum(wt * ctx.p[:, min(y + tau, ny - 1)] for tau, wt in zip(C.TAU, C.PTAU))
    return _clip(ctx, ctx.pstar * target ** gamma)


RULE_FUNCS = {
    "gleich": rule_gleich, "folgend": rule_folgend, "anti": rule_anti, "voraus": rule_voraus,
    "nach": rule_nach, "lueck": rule_lueck, "perfekt": rule_perfekt,
}


def make_rule(name: str, gamma: float):
    """Bindet die Stärke gamma an die Regel; liefert die Funktion (y, ctx) für `simulate`."""
    fn = RULE_FUNCS[name]
    return lambda y, ctx: fn(y, ctx, gamma)
