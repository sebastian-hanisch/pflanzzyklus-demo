"""Marktmodell mit Pflanzverzug (Spinnweb): Der Preis heute folgt dem Angebot aus Pflanzungen von vor 7 bis 12 Jahren.

ln p_t = -phi * ln S_t + Nachfragestoß,  S_t = sum_k w_k * Q_(t-k) * Angebotsstoß,  Q_y = p_y^eta (begrenzt).
Der Hof ist Preisnehmer: sein Verhalten wirkt nicht auf den Markt.
"""

import numpy as np

import pfz_constants as C


def market_paths(n: int, seed: int, phi: float, eta: float = C.ETA, years: int | None = None):
    """Preisindex p [n, years] (je Pfad im Mittel 1) und gepflanzte Marktmenge Q [n, years] für die Jahre PRE + H.

    Der Pfad läuft C.WARM Jahre ein; die Anfangsstörung stößt den Zyklus an.
    """
    years = years or (C.PRE + C.H)
    rng = np.random.default_rng(seed)
    total = C.WARM + years
    maxlag = max(C.TAU) + 1
    q = np.ones((n, total + maxlag))
    p = np.ones((n, total))
    q[:, : maxlag + 2] = np.exp(0.15 * rng.standard_normal((n, maxlag + 2)))
    for t in range(total):
        s = np.zeros(n)
        for tau, w in zip(C.TAU, C.PTAU):
            s += w * q[:, maxlag + t - tau]
        s *= np.exp(C.SIG_MARKET_SUPPLY * rng.standard_normal(n) - C.SIG_MARKET_SUPPLY**2 / 2)
        p[:, t] = np.exp(-phi * np.log(s) + C.SIG_MARKET_DEMAND * rng.standard_normal(n))
        q[:, maxlag + t] = np.clip(p[:, t] ** eta, *C.Q_CAP)
    p = p[:, C.WARM:]
    q = q[:, maxlag + C.WARM: maxlag + total]
    return p / p.mean(axis=1, keepdims=True), q


def price_spread(p: np.ndarray, skip: int = 20) -> float:
    """Streuung von ln p über die Zeit, gemittelt über die Pfade (ab Jahr `skip`)."""
    return float(np.log(p[:, skip:]).std(axis=1).mean())
