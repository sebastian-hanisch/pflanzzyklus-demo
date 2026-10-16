"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Regeln und Presets.

Alle Eingangszahlen sind Annahmen oder Verbandsrichtwerte (siehe README, Abschnitt "Ehrliche Grenzen"), keine Messung aus einem Betrieb.
"""

# --- Akt 1: Pflanzzyklus -----------------------------------------------------------------------------------------------------------
# Ankunft der Verkaufsgröße (1,5 m) nach TAU Jahren ab Pflanzung (je Baum), plus Verschiebung je Pflanzjahr (Standort, Wetter).
TAU = (7, 8, 9, 10, 11, 12)
PTAU = (0.05, 0.20, 0.30, 0.25, 0.15, 0.05)
SHIFT = (-1, 0, 1)
PSHIFT = (0.25, 0.50, 0.25)
PRE = 15                      # Vorgeschichte in Jahren (gleich für alle Regeln, Pflanzung nach Sollmenge)
H = 90                        # Simulationsjahre nach der Vorgeschichte
BURN = 22                     # ausgewertet wird ab diesem Jahr (Einschwingen)
WARM = 80                     # Einlaufjahre des Marktmodells
N_PATHS = 300                 # Marktpfade je Auswertung (Komplexität: O(N_PATHS * (PRE + H) * Regeln))
N_PATHS_SWEEP = 300
SEED_MAX, DEFAULT_SEED = 999999, 7
SWEEP_SEEDS = (100000, 100001, 100002)   # feste Pfadmengen für das Experiment: nie ein einzelner Seed als Befund

PHI_MIN, PHI_MAX, DEFAULT_PHI = 0.5, 1.4, 1.1
PHI_SWEEP = (0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4)
ETA = 1.0                     # Pflanzreaktion des Marktes auf den Preis (fest; Regler ist das Produkt phi * eta)
SIG_MARKET_DEMAND, SIG_MARKET_SUPPLY = 0.10, 0.10
Q_CAP = (0.3, 2.5)            # Grenzen der Marktpflanzung (relativ zum Mittel)

EXACT_HORIZON = 40            # Pflanzjahre im LP der exakten Referenz
EXACT_PATHS = 10              # Pfade, über die der Abstand zum Optimum gemittelt wird

GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = 0.25, 2.0, 1.0
DRY_MIN, DRY_MAX, DEFAULT_DRY = 0, 40, 15       # Anteil Dürrejahre [%]
WINDOW_MIN, WINDOW_MAX, DEFAULT_WINDOW = 1, 5, 3  # Jahre, in denen ein Baum nach Erreichen der Größe verkauft werden kann
DEMAND_MIN, DEMAND_MAX, DEFAULT_DEMAND = 1000, 4000, 2000   # Absatz des Hofs [Bäume je Jahr]

FARM_DEFAULTS = dict(
    sig_d=0.20,               # Streuung der Jahresnachfrage (lognormal)
    s0=0.88,                  # Überleben nach der Pflanzung in normalen Jahren
    dry_lo=0.20, dry_hi=0.60,  # Zusatzausfall im Dürrejahr (gleichverteilt)
    q_a=0.80,                 # Anteil Verkaufsware unter den Überlebenden
    price=45.0,               # EUR je Baum bei Preisindex 1 (1,75 m x 26 EUR/m)
    c_plant=1.60, c_care=1.50,  # EUR je Baum: Pflanzung; Pflege je Baum und Jahr
    pmax_factor=1.5,          # Flächengrenze: höchstens so viel mal die Sollpflanzmenge je Jahr
)
RULE_CLIP = (0.3, 2.0)        # Pflanzmenge der preisabhängigen Regeln relativ zur Sollmenge
LUECKEN_BUFFER = 0.0

RULES = ("gleich", "folgend", "anti", "voraus", "nach", "lueck", "perfekt")
RULE_NAME = {
    "gleich": "Gleichmäßig pflanzen",
    "folgend": "Dem Preis folgen",
    "anti": "Antizyklisch pflanzen",
    "voraus": "Vorausschau (öffentliche Pflanzzahlen)",
    "nach": "Ausfall nachpflanzen",
    "lueck": "Lückenfüller",
    "perfekt": "Preis in 9 Jahren bekannt (Referenz)",
}
RULE_TEXT = {
    "gleich": "Jedes Jahr die Sollmenge, die bei durchschnittlichem Ausfall genau den Absatz deckt. Reagiert auf nichts.",
    "folgend": "Pflanzmenge = Sollmenge mal Preisindex hoch Stärke: bei hohem Preis mehr, bei niedrigem weniger. So handelt der Markt als Ganzes.",
    "anti": "Pflanzmenge = Sollmenge mal Preisindex hoch minus Stärke: bei niedrigem Preis mehr, in der Hoffnung, dass der Preis in zehn Jahren oben ist.",
    "voraus": "Rechnet aus den Pflanzmengen des Marktes in den Vorjahren den Preis in neun Jahren aus (die Zukunft der nächsten zwei Jahre mit dem "
              "Mittel) und pflanzt danach: mehr, wenn dann ein hoher Preis zu erwarten ist.",
    "nach": "Sollmenge plus Ersatz für den Ausfall des Vorjahres: Was im Dürrejahr verloren ging, wird im Folgejahr zusätzlich gepflanzt.",
    "lueck": "Schaut auf die erwarteten Ankünfte aus allen schon gepflanzten Jahrgängen für die Jahre 8 bis 10 und pflanzt genau den fehlenden "
             "Rest zum Absatz.",
    "perfekt": "Referenz, keine Obergrenze: kennt den tatsächlichen Preisindex der nächsten Jahre (gewichtet nach der Ankunftszeit) und "
               "pflanzt danach. Zeigt, was Wissen über die Zukunft wert ist.",
}

# --- Akt 2: Saison -------------------------------------------------------------------------------------------------------------------
S0 = 2000                     # Bäume zu Saisonbeginn
SEASON_START = (2026, 11, 27)  # Tag 1 = Freitag; 27 Tage bis 23.12.2026
SEASON_DAYS = 27
SEASON_N_TRAIN, SEASON_N_TEST = 300, 600   # Saisons zum Abstimmen bzw. Auswerten (Komplexität O(Saisons * 27 * 30))
SEASON_PRICE_GRID = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
SEASON_LOAD_GRID = (0, 10, 20, 30, 45, 60, 80, 110, 150, 200)
SEASON_DEFAULTS = dict(
    p_hof=45.0, p_markt=45.0,  # EUR je Baum bei Preisfaktor 1
    c_cut=2.5, c_carry=0.8,   # Schlagen; Mitführen auf den Markt (Laden, Transport, Rückladen)
    v_carry=0.4,              # Wert eines stehenden Restbaums im Folgejahr (Anteil des Hofpreises)
    sig_w=0.35, sig_z=0.15,   # Tagesfaktor (Wetter) und Saisonfaktor der Nachfrage
)
RHO_MIN, RHO_MAX, DEFAULT_RHO = 0.6, 1.5, 1.0
KAPPA_MIN, KAPPA_MAX, DEFAULT_KAPPA = 0.5, 3.0, 1.5
SHARE_MIN, SHARE_MAX, DEFAULT_SHARE = 0, 50, 35     # Anteil des Marktes an der Gesamtnachfrage [%]
FEE_MIN, FEE_MAX, DEFAULT_FEE = 0, 600, 140         # Kosten je Markttag [EUR]
DECAY_MIN, DECAY_MAX, DEFAULT_DECAY = 0, 30, 6      # Wertverlust geschlagener Bäume je Tag nach 2 Tagen [%]
BACK_OPTIONS = ("ja", "nein")

COLORS = {
    "gleich": "#2F6B65", "folgend": "#C0504D", "anti": "#E0A030", "voraus": "#4c78a8", "nach": "#8E6BB0", "lueck": "#7C8B3A",
    "perfekt": "#555555",
}
COLOR_PRICE = "#4c78a8"
COLOR_GOOD, COLOR_BAD, COLOR_NEUTRAL = "#2F6B65", "#C0504D", "rgba(150,150,150,0.7)"

PRESETS = {
    "Stabiler Markt": dict(phi=0.8, gamma=DEFAULT_GAMMA, dry=DEFAULT_DRY, window=DEFAULT_WINDOW, demand=DEFAULT_DEMAND, seed=DEFAULT_SEED,
                           share=DEFAULT_SHARE, fee=DEFAULT_FEE, decay=DEFAULT_DECAY, rho=DEFAULT_RHO, kappa=DEFAULT_KAPPA, back="ja"),
    "Mittlere Schwankung (Voreinstellung)": dict(phi=DEFAULT_PHI, gamma=DEFAULT_GAMMA, dry=DEFAULT_DRY, window=DEFAULT_WINDOW,
                                                 demand=DEFAULT_DEMAND, seed=DEFAULT_SEED, share=DEFAULT_SHARE, fee=DEFAULT_FEE,
                                                 decay=DEFAULT_DECAY, rho=DEFAULT_RHO, kappa=DEFAULT_KAPPA, back="ja"),
    "Starker Zyklus": dict(phi=1.3, gamma=DEFAULT_GAMMA, dry=DEFAULT_DRY, window=DEFAULT_WINDOW, demand=DEFAULT_DEMAND, seed=DEFAULT_SEED,
                           share=DEFAULT_SHARE, fee=DEFAULT_FEE, decay=DEFAULT_DECAY, rho=DEFAULT_RHO, kappa=DEFAULT_KAPPA, back="ja"),
    "Viele Dürrejahre": dict(phi=DEFAULT_PHI, gamma=DEFAULT_GAMMA, dry=35, window=DEFAULT_WINDOW, demand=DEFAULT_DEMAND, seed=DEFAULT_SEED,
                             share=DEFAULT_SHARE, fee=DEFAULT_FEE, decay=DEFAULT_DECAY, rho=DEFAULT_RHO, kappa=DEFAULT_KAPPA, back="ja"),
    "Hof ohne Rücknahme": dict(phi=DEFAULT_PHI, gamma=DEFAULT_GAMMA, dry=DEFAULT_DRY, window=DEFAULT_WINDOW, demand=DEFAULT_DEMAND,
                               seed=DEFAULT_SEED, share=DEFAULT_SHARE, fee=DEFAULT_FEE, decay=15, rho=DEFAULT_RHO, kappa=DEFAULT_KAPPA,
                               back="nein"),
}
PRESET_HELP = {
    "Stabiler Markt": "Der Markt reagiert schwach auf Preise (Marktreaktion 0,8): der Preis schwankt wenig, die Pflanzregel ist fast egal.",
    "Mittlere Schwankung (Voreinstellung)": "Marktreaktion 1,1, knapp an der Schwelle zum Schwingen: der Preis schwankt mäßig.",
    "Starker Zyklus": "Marktreaktion 1,3: Überangebot und Knappheit lösen sich über viele Jahre ab; wer dem Preis folgt, pflanzt im Takt des Marktes.",
    "Viele Dürrejahre": "35 % der Jahre sind Dürrejahre mit hohem Ausfall der Neupflanzung; zeigt, ob Nachpflanzen oder Lückenfüller etwas bringen.",
    "Hof ohne Rücknahme": "Akt 2: der Hof kann geschlagene Rückläufer vom Markt nicht selbst verkaufen (reiner Selbstschlag), Wertverlust 15 % je Tag.",
}
