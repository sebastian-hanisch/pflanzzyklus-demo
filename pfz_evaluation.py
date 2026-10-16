"""Auswertung: Einstellungen, Mehr-Pfad-Vergleich der Pflanzregeln, Experiment über die Marktreaktion, exakte Referenz, Saison-Hebel, Verdikte.

Verdikte sind reine Funktionen der berechneten Werte: jede Zahl im Text kommt aus der Rechnung (DEMO-PLAYBOOK §1d).
"""

from dataclasses import dataclass, field

import numpy as np

import pfz_constants as C
import pfz_orchard as O
import pfz_season as S
from pfz_exact import PlanInstance, evaluate_plan, solve_exact
from pfz_market import market_paths, price_spread
from pfz_rules import make_rule


def fmt(x, digits=1):
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x, digits=1, sign=False):
    return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace(".", ",") + " %"


def eur(x):
    return f"{x:,.0f}".replace(",", ".") + " EUR"


@dataclass(frozen=True)
class OrchardSettings:
    phi: float = C.DEFAULT_PHI
    gamma: float = C.DEFAULT_GAMMA
    dry: int = C.DEFAULT_DRY
    window: int = C.DEFAULT_WINDOW
    demand: int = C.DEFAULT_DEMAND
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class SeasonSettings:
    share: int = C.DEFAULT_SHARE
    fee: int = C.DEFAULT_FEE
    decay: int = C.DEFAULT_DECAY
    rho: float = C.DEFAULT_RHO
    kappa: float = C.DEFAULT_KAPPA
    back: str = "ja"

    def params(self) -> S.SeasonParams:
        return S.SeasonParams(rho=self.rho, kappa=self.kappa, share=self.share / 100.0, fee=float(self.fee),
                              decay=self.decay / 100.0, back=self.back == "ja")


@dataclass(frozen=True)
class Settings:
    phi: float = C.DEFAULT_PHI
    gamma: float = C.DEFAULT_GAMMA
    dry: int = C.DEFAULT_DRY
    window: int = C.DEFAULT_WINDOW
    demand: int = C.DEFAULT_DEMAND
    seed: int = C.DEFAULT_SEED
    share: int = C.DEFAULT_SHARE
    fee: int = C.DEFAULT_FEE
    decay: int = C.DEFAULT_DECAY
    rho: float = C.DEFAULT_RHO
    kappa: float = C.DEFAULT_KAPPA
    back: str = "ja"

    def orchard(self) -> OrchardSettings:
        return OrchardSettings(self.phi, self.gamma, self.dry, self.window, self.demand, self.seed)

    def season(self) -> SeasonSettings:
        return SeasonSettings(self.share, self.fee, self.decay, self.rho, self.kappa, self.back)


# --- Akt 1 ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class RuleStats:
    profit: float           # Jahresüberschuss im Mittel über die Pfade [EUR]
    delta: float            # Differenz zu "gleichmäßig" [EUR], gepaart
    delta_pct: float        # in % des Überschusses von "gleichmäßig"
    delta_se_pct: float     # Standardfehler dieser Differenz [Prozentpunkte]
    lost: float             # entgangene Verkäufe je Jahr
    waste: float            # verworfene (zu hohe) Bäume je Jahr
    planted: float          # gepflanzte Bäume je Jahr


@dataclass
class Analysis:
    settings: OrchardSettings
    farm: O.Farm
    spread: float
    stats: dict
    timeline: dict          # Pfad 0: Preisindex und Pflanzmengen je Regel
    n_paths: int
    problems: list          # Verletzungen des unabhängigen Prüfers (leer = in Ordnung)


def build_paths(os_: OrchardSettings, n: int, market_seed: int, shock_seed: int):
    farm = O.make_farm(os_.demand, os_.window, os_.dry)
    p, q = market_paths(n, market_seed, os_.phi)
    return farm, O.draw_shocks(farm, p, q, shock_seed), p


def run_rules(farm: O.Farm, shocks: O.Shocks, phi: float, gamma: float, rules=C.RULES) -> dict:
    """Alle Regeln auf denselben Pfaden. Gibt {Regel: Zusammenfassung (summarize) + 'planted_path0'} zurück."""
    out = {}
    for r in rules:
        res = O.simulate(farm, shocks, make_rule(r, gamma), phi)
        s = O.summarize(res)
        s["planted_path0"] = res.planted[0, C.PRE:].copy()
        s["problems"] = O.check_result(farm, shocks, res)
        out[r] = s
    return out


def rule_stats(summaries: dict) -> dict:
    base = summaries["gleich"]["profit"]
    n = len(base)
    stats = {}
    for r, s in summaries.items():
        d = s["profit"] - base
        stats[r] = RuleStats(
            profit=float(s["profit"].mean()), delta=float(d.mean()), delta_pct=float(d.mean() / base.mean() * 100),
            delta_se_pct=float(d.std(ddof=1) / np.sqrt(n) / base.mean() * 100) if r != "gleich" else 0.0,
            lost=float(s["lost"].mean()), waste=float(s["waste"].mean()), planted=float(s["planted"].mean()))
    return stats


def analyse(os_: OrchardSettings) -> Analysis:
    farm, shocks, p = build_paths(os_, C.N_PATHS, os_.seed, os_.seed + 1)
    summaries = run_rules(farm, shocks, os_.phi, os_.gamma)
    timeline = dict(price=p[0, C.PRE:].copy(), planted={r: summaries[r]["planted_path0"] for r in C.RULES})
    problems = [f"{C.RULE_NAME[r]}: {msg}" for r in C.RULES for msg in summaries[r]["problems"]]
    return Analysis(os_, farm, price_spread(p), rule_stats(summaries), timeline, C.N_PATHS, problems)


def sweep_phi(os_: OrchardSettings) -> list:
    """Experiment: Marktreaktion phi von schwach bis stark, jede Zelle über die festen Pfadmengen SWEEP_SEEDS (je ein Drittel der Pfade)."""
    farm = O.make_farm(os_.demand, os_.window, os_.dry)
    rows = []
    per = C.N_PATHS_SWEEP // len(C.SWEEP_SEEDS)
    for phi in C.PHI_SWEEP:
        parts = {r: [] for r in C.RULES}
        spreads = []
        for sd in C.SWEEP_SEEDS:
            p, q = market_paths(per, sd, phi)
            shocks = O.draw_shocks(farm, p, q, sd + 1)
            spreads.append(price_spread(p))
            for r, s in run_rules(farm, shocks, phi, os_.gamma).items():
                parts[r].append(s["profit"])
        summ = {r: dict(profit=np.concatenate(v), lost=np.zeros(1), waste=np.zeros(1), planted=np.zeros(1)) for r, v in parts.items()}
        st = rule_stats(summ)
        rows.append(dict(phi=phi, spread=float(np.mean(spreads)), delta={r: st[r].delta_pct for r in C.RULES},
                         se={r: st[r].delta_se_pct for r in C.RULES}, profit=st["gleich"].profit))
    return rows


def verdict_core(a: Analysis):
    s = a.stats
    fol = s["folgend"]
    others = [r for r in C.RULES if r not in ("gleich", "folgend", "perfekt")]
    best = max(others, key=lambda r: s[r].delta_pct)
    tol = max(1.0, 2 * fol.delta_se_pct)
    spread = fmt(a.spread, 2)
    if fol.delta_pct < -tol:
        head = (f"Dem Preis hinterherzupflanzen kostet bei einer Preisstreuung von {spread} (Streuung von ln p) {pct(-fol.delta_pct)} des "
                f"Jahresüberschusses gegenüber gleichmäßigem Pflanzen: die Bäume werden erst in 7 bis 12 Jahren verkauft, dann hat der Markt gekippt.")
        kind = "warning"
    elif fol.delta_pct > tol:
        head = f"Dem Preis zu folgen bringt hier {pct(fol.delta_pct)} gegenüber gleichmäßigem Pflanzen: bei so schwacher Marktreaktion lohnt sich die Anpassung."
        kind = "success"
    else:
        head = (f"Bei einer Preisstreuung von {spread} ist es fast gleichgültig, ob Sie dem Preis folgen ({pct(fol.delta_pct, sign=True)}, "
                f"Standardfehler {fmt(fol.delta_se_pct)} Prozentpunkte): der Markt schwankt zu wenig, um die Regel zu bestrafen.")
        kind = "info"
    b = s[best]
    if b.delta_pct > max(1.0, 2 * b.delta_se_pct):
        tail = f" Beste Alternative: {C.RULE_NAME[best]} mit {pct(b.delta_pct, sign=True)}."
    else:
        tail = (f" Keine der übrigen Regeln (Vorausschau, Nachpflanzen, Lückenfüller, antizyklisch) schlägt gleichmäßiges Pflanzen um mehr als "
                f"{fmt(max(1.0, 2 * b.delta_se_pct), 0)} %; beste: {C.RULE_NAME[best]} mit {pct(b.delta_pct, sign=True)}.")
    return kind, head + tail


def verdict_sweep(rows: list):
    """Ab welcher Marktreaktion kostet das Folgen des Preises spürbar (mehr als 2 % und mehr als zwei Standardfehler)?"""
    hit = [r for r in rows if r["delta"]["folgend"] < -max(2.0, 2 * r["se"]["folgend"])]
    last = rows[-1]
    if not hit:
        return "info", (f"In keiner der gerechneten Marktreaktionen (bis {fmt(last['phi'])}) kostet das Folgen des Preises mehr als 2 %.")
    first = hit[0]
    return "warning", (f"Das Folgen des Preises kostet ab einer Marktreaktion von {fmt(first['phi'])} (Preisstreuung {fmt(first['spread'], 2)}) mehr als 2 % "
                       f"des Jahresüberschusses ({pct(-first['delta']['folgend'])}); bei {fmt(last['phi'])} (Preisstreuung {fmt(last['spread'], 2)}) sind es "
                       f"{pct(-last['delta']['folgend'])}. Vergleichsgröße ist gleichmäßiges Pflanzen.")


# --- Exakte Referenz (HiGHS) -------------------------------------------------------------------------------------------------------------

@dataclass
class ExactResult:
    n_paths: int
    gap_mean: dict          # Regel -> Abstand zum Optimum in % des Optimums, gemittelt über die Pfade
    optimum_vs_gleich: float  # Optimum gegen "gleichmäßig" in %
    optimum_vs_gleich_se: float
    path0: dict             # Pfad 0: Preis, Plan des LP, Pläne der Regeln
    horizon: int


def exact_comparison(os_: OrchardSettings, n_paths: int = C.EXACT_PATHS) -> ExactResult:
    farm, shocks, p = build_paths(os_, n_paths, os_.seed, os_.seed + 1)
    det = O.Shocks(shocks.n, np.where(np.arange(shocks.surv.shape[1])[None, :] < C.PRE, shocks.surv, O.expected_survival(farm)),
                   shocks.shift, np.ones_like(shocks.eps_d), shocks.p, shocks.q, shocks.seed)
    hl = C.EXACT_HORIZON
    rules = ("gleich", "folgend", "voraus", "nach", "lueck", "perfekt")
    plans = {r: O.simulate(farm, det, make_rule(r, os_.gamma), os_.phi, expected=True).planted[:, C.PRE:C.PRE + hl] for r in rules}
    gaps = {r: [] for r in rules}
    opt_vs_gleich = []
    path0 = {}
    for i in range(n_paths):
        hist = det.surv[i, :C.PRE] * O.planting_target(farm) * farm.q_a
        inst = PlanInstance(farm, det.p[i, C.PRE:C.PRE + hl + len(O.PI) + farm.window], hist, hl)
        opt = solve_exact(inst)
        vals = {r: solve_exact(inst, fixed_planting=plans[r][i]).objective for r in rules}
        for r in rules:
            gaps[r].append((opt.objective - vals[r]) / opt.objective * 100)
        opt_vs_gleich.append((opt.objective / vals["gleich"] - 1) * 100)
        if i == 0:
            path0 = dict(price=inst.price.copy(), optimum=opt.planting.copy(), plans={r: plans[r][0].copy() for r in rules},
                         values={r: vals[r] for r in rules}, objective=opt.objective)
    ov = np.array(opt_vs_gleich)
    return ExactResult(n_paths, {r: float(np.mean(g)) for r, g in gaps.items()}, float(ov.mean()),
                       float(ov.std(ddof=1) / np.sqrt(len(ov))) if len(ov) > 1 else 0.0, path0, hl)


# --- Akt 2: Saison -----------------------------------------------------------------------------------------------------------------------

@dataclass
class Lever:
    key: str
    name: str
    value: float            # EUR je Saison
    se: float
    pct: float              # in % des Gewinns ohne Markt bei Richtpreis


@dataclass
class SeasonAnalysis:
    settings: SeasonSettings
    base: float             # Gewinn je Saison: Richtpreis, kein Markt
    levers: list
    load_best: int
    m_best: float
    details: dict


def _paired(a, b):
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def analyse_season(ss: SeasonSettings) -> SeasonAnalysis:
    sp = ss.params()
    u_tr = S.draw_factors(sp, C.SEASON_N_TRAIN, 11)
    u_te = S.draw_factors(sp, C.SEASON_N_TEST, 22)
    dr_tr = S.draw_demand(sp, u_tr, 3, 1.0)
    dr = S.draw_demand(sp, u_te, 9, 1.0)
    sim = S.simulate_season
    tuned = {L: sim(sp, u_tr, 5, S.price_fixed(1.0), S.markt_fixed(L), fixed_draws=dr_tr)[0].mean() for L in C.SEASON_LOAD_GRID}
    load_best = max(tuned, key=tuned.get)
    kein, d_kein = sim(sp, u_te, 7, S.price_fixed(1.0), S.markt_none, fixed_draws=dr)
    fest, d_fest = sim(sp, u_te, 7, S.price_fixed(1.0), S.markt_fixed(load_best), fixed_draws=dr)
    value1, _ = S.dp_prices(sp, grid=(1.0,))
    nv, d_nv = sim(sp, u_te, 7, S.price_fixed(1.0), S.markt_newsvendor(sp, value1), fixed_draws=dr)
    hind = S.hindsight(sp, u_te, 1.0, dr)
    # Preis, nur Hofverkauf
    by_price = {m: sim(sp, u_tr, 5, S.price_fixed(m), S.markt_none)[0].mean() for m in C.SEASON_PRICE_GRID}
    m_best = max(by_price, key=by_price.get)
    p1 = sim(sp, u_te, 8, S.price_fixed(1.0), S.markt_none)[0]
    pb = sim(sp, u_te, 8, S.price_fixed(m_best), S.markt_none)[0]
    value, policy = S.dp_prices(sp)
    pd_, d_dp = sim(sp, u_te, 8, S.price_dp(policy), S.markt_none)
    base = float(kein.mean())
    items = [
        ("markt", "Markt als zweiter Absatzweg (feste Menge je Markttag)", *_paired(fest, kein)),
        ("regel", "Marktmenge nach Schattenpreis statt fester Menge", *_paired(nv, fest)),
        ("preis", "Bester fester Preis statt Richtpreis (nur Hof)", *_paired(pb, p1)),
        ("preisplan", "Preis nach Tag und Restbestand statt bestem festem Preis (nur Hof)", *_paired(pd_, pb)),
        ("unkenntnis", "Preis der Unkenntnis: Vorabkenntnis der Nachfrage statt Schattenpreisregel", *_paired(hind, nv)),
    ]
    levers = [Lever(k, n, v, se, v / base * 100) for k, n, v, se in items]
    details = dict(qloss_nv=float(d_nv["qloss"].mean()), qloss_fest=float(d_fest["qloss"].mean()), attended_nv=float(d_nv["attended"].mean()),
                   sold_m_nv=float(d_nv["sold_m"].mean()), left_nv=float(d_nv["left"].mean()), left_kein=float(d_kein["left"].mean()),
                   price_dp_left=float(d_dp["left"].mean()), nv=float(nv.mean()), fest=float(fest.mean()), hind=float(hind.mean()),
                   p_best=float(pb.mean()), p_dp=float(pd_.mean()), p_one=float(p1.mean()), n_markt_days=S.N_MARKT_DAYS)
    return SeasonAnalysis(ss, base, levers, int(load_best), float(m_best), details)


def verdict_season(a: SeasonAnalysis):
    lv = {l.key: l for l in a.levers}
    small = [l for l in a.levers if l.key != "markt"]
    biggest = max(small, key=lambda l: abs(l.pct))
    ss = a.settings
    text = (f"Der Markt als zweiter Absatzweg bringt {pct(lv['markt'].pct, sign=True)} ({eur(lv['markt'].value)} je Saison); das folgt aus dem "
            f"angenommenen Marktanteil von {ss.share} % der Nachfrage. Alle übrigen Hebel sind kleiner: der größte ist "
            f"„{biggest.name}“ mit {pct(biggest.pct, sign=True)}.")
    if abs(biggest.pct) < 3:
        return "info", text + " Die Saison ist bei dieser Einstellung fast gelöst: keine Regel und kein Preisplan ändert den Gewinn um mehr als 3 %."
    return "warning", text + " Bei dieser Einstellung lohnt es sich, die Regel zu prüfen."


def verdict_exact(ex: ExactResult):
    """Text zur exakten Referenz: Abstand der Regeln zum Optimum im Erwartungswert (mit bekanntem Preispfad)."""
    g = ex.gap_mean
    best = min((r for r in g if r != "perfekt"), key=lambda r: g[r])
    if best == "gleich":
        near = f"Am nächsten am Optimum liegt gleichmäßiges Pflanzen ({pct(g['gleich'])} darunter), dem Preis zu folgen liegt {pct(g['folgend'])} darunter. "
    else:
        near = (f"Am nächsten am Optimum liegt {C.RULE_NAME[best]} ({pct(g[best])} darunter), gleichmäßig pflanzen liegt {pct(g['gleich'])} darunter, "
                f"dem Preis zu folgen {pct(g['folgend'])}. ")
    return "info", (f"Im Erwartungswert (Ausfall gemittelt, Nachfrage fest, Preispfad bekannt) liegt das Optimum im Mittel über {ex.n_paths} Pfade "
                    f"{pct(ex.optimum_vs_gleich, sign=True)} (Standardfehler {fmt(ex.optimum_vs_gleich_se)} Prozentpunkte) über gleichmäßigem Pflanzen. "
                    + near +
                    "Das Optimum kennt den ganzen Preispfad im Voraus; es ist eine Obergrenze nur für diese Erwartungswert-Rechnung, kein Plan für den zufälligen Betrieb.")


def season_notes(sa: SeasonAnalysis) -> list:
    """Hinweise zu Nullspalten und Bedingungen, aus den berechneten Werten gebildet."""
    d, ss = sa.details, sa.settings
    notes = []
    lv = {l.key: l for l in sa.levers}
    if abs(lv["preisplan"].value) < 1e-9:
        notes.append(f"Der Preis nach Tag und Restbestand ist hier derselbe wie der beste feste Preis ({fmt(sa.m_best, 2)} mal Richtpreis): "
                     f"der Bestand wird nie knapp (nach der Saison bleiben im Mittel {fmt(d['price_dp_left'], 0)} von {C.S0} Bäumen), "
                     "also hat der Plan nichts zu steuern. Bei knappem Bestand (Gesamtnachfrage 1,3 mal Bestand, starke Preisreaktion) weicht er ab.")
    notes.append(f"Wertverlust der Rückläufer im Mittel: {eur(d['qloss_nv'])} je Saison bei der Schattenpreisregel, {eur(d['qloss_fest'])} bei der "
                 f"festen Menge ({ss.decay} % je Tag nach zwei Tagen, Hof nimmt Rückläufer {'zurück' if ss.back == 'ja' else 'nicht zurück'}).")
    notes.append(f"Der Hof besucht den Markt im Mittel an {fmt(d['attended_nv'])} von {d['n_markt_days']} Markttagen und verkauft dort "
                 f"{fmt(d['sold_m_nv'], 0)} Bäume; am Saisonende stehen noch {fmt(d['left_nv'], 0)} Bäume (ohne Markt {fmt(d['left_kein'], 0)}).")
    return notes
