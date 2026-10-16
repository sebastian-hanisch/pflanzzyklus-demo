"""Pflanzzyklus: Wie viel pflanzen, wenn der Verkauf zehn Jahre später ist? - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Akt 1: Ein Weihnachtsbaumhof pflanzt heute Bäume, die erst in 7 bis 12 Jahren verkauft werden. Sieben Pflanzregeln laufen auf denselben
Marktpfaden mit Pflanzverzug (Spinnweb); eine exakte Referenz (HiGHS) zeigt, was Wissen über den Preis wert wäre.
Akt 2: ein kurzer Hebel-Check der Saison (Hofverkauf und Wochenmarkt): wo steckt in den 27 Verkaufstagen überhaupt noch etwas zu holen?

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import pfz_constants as C
from pfz_evaluation import (
    Settings,
    analyse,
    analyse_season,
    eur,
    exact_comparison,
    fmt,
    pct,
    season_notes,
    sweep_phi,
    verdict_core,
    verdict_exact,
    verdict_season,
    verdict_sweep,
)
from pfz_pdf_export import build_pdf
from pfz_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pfz_visualization import build_exact, build_levers, build_rule_bars, build_sweep, build_timeline

st.set_page_config(page_title="Pflanzzyklus – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(orchard):
    return analyse(orchard)


@st.cache_data(show_spinner=False)
def _sweep(orchard):
    return sweep_phi(orchard)


@st.cache_data(show_spinner=False)
def _exact(orchard):
    return exact_comparison(orchard)


@st.cache_data(show_spinner=False)
def _season(season):
    return analyse_season(season)


st.title("🌲 Pflanzzyklus: Wie viel pflanzen, wenn der Verkauf zehn Jahre später ist?")
st.markdown(
    """
Ein Weihnachtsbaum, den Sie heute pflanzen, wird erst in **7 bis 12 Jahren** verkauft. In **Akt 1** läuft ein kleiner Hof (rund 2 000 Bäume
im Jahr) über 90 Jahre und **sieben Pflanzregeln** konkurrieren auf denselben **Marktpfaden mit Pflanzverzug**: Wenn alle bei hohem Preis
pflanzen, ist der Markt in zehn Jahren überfüllt (Spinnweb-Modell). Dazu **Dürrejahre** mit hohem Ausfall und gestreutes Wachstum.
**Akt 2** ist ein kurzer **Hebel-Check der Saison** (Hofverkauf und Wochenmarkt, 27 Tage): Wo steckt dort überhaupt noch etwas zu holen?

Weiter unten: die **Kernfrage**, ein **Experiment** über die Marktreaktion, **Akt 2** und der Expander **🔧 Wie wir das erreichen – vollständiger
Methodenvergleich** mit den Regeln und der **🧮 exakten Referenz**.
"""
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Akt 1: Markt und Hof**")
    phi = st.slider("Marktreaktion", *bounds("phi_slider"), step=0.05, key="phi_slider",
                    help="Preisflexibilität des Marktes mal Pflanzreaktion der Betriebe. Ab etwa 1,1 schwingt der Markt, je höher, desto stärker.")
    gamma = st.slider("Stärke der Regeln", *bounds("gamma_slider"), step=0.25, key="gamma_slider",
                      help="Wie stark die preisabhängigen Regeln die Pflanzmenge anpassen (Preisindex hoch Stärke).")
    dry = st.slider("Dürrejahre [%]", *bounds("dry_slider"), step=5, key="dry_slider",
                    help="Anteil der Jahre, in denen die Neupflanzung 20 bis 60 % zusätzlich ausfällt.")
    window = st.slider("Verkaufsfenster [Jahre]", *bounds("window_slider"), step=1, key="window_slider",
                       help="So viele Jahre kann ein Baum nach Erreichen der Verkaufsgröße noch verkauft werden, danach ist er zu hoch.")
    demand = st.slider("Absatz des Hofs [Bäume je Jahr]", *bounds("demand_slider"), step=250, key="demand_slider",
                       help="Feste Absatzmenge (Hof und Markt zusammen), streut von Jahr zu Jahr um etwa 20 %.")
    st.markdown("**Akt 2: Saison**")
    share = st.slider("Marktanteil der Nachfrage [%]", *bounds("share_slider"), step=5, key="share_slider",
                      help="Anteil der Gesamtnachfrage, der auf dem Wochenmarkt entsteht (Mittwoch und Samstag).")
    fee = st.slider("Kosten je Markttag [EUR]", *bounds("fee_slider"), step=20, key="fee_slider", help="Standgebühr und Arbeitszeit je Markttag.")
    decay = st.slider("Wertverlust geschlagener Bäume [% je Tag]", *bounds("decay_slider"), step=1, key="decay_slider",
                      help="Nach zwei Tagen Schnittalter verliert ein geschlagener Baum so viel seines Wertes je Tag.")
    rho = st.slider("Gesamtnachfrage / Bestand", *bounds("rho_slider"), step=0.1, key="rho_slider",
                    help="Erwartete Kundenzahl der Saison beim Richtpreis, geteilt durch die 2 000 Bäume im Bestand.")
    kappa = st.slider("Preisreaktion der Kunden", *bounds("kappa_slider"), step=0.5, key="kappa_slider",
                      help="Mengenverlust je Preiserhöhung: 1,5 heißt 15 % weniger Kunden bei 10 % höherem Preis.")
    back = st.radio("Hof nimmt Rückläufer zurück", C.BACK_OPTIONS, key="back_radio", horizontal=True,
                    help="Ja: geschlagene, auf dem Markt nicht verkaufte Bäume gehen in den Hofverkauf. Nein: reiner Selbstschlag, nur der Markt kann sie noch verkaufen.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Marktpfade", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Marktpfade generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"phi_slider": float(phi), "gamma_slider": float(gamma), "dry_slider": int(dry), "window_slider": int(window),
                   "demand_slider": int(demand), "seed_input": int(seed), "share_slider": int(share), "fee_slider": int(fee),
                   "decay_slider": int(decay), "rho_slider": float(rho), "kappa_slider": float(kappa), "back_radio": back})

settings = Settings(float(phi), float(gamma), int(dry), int(window), int(demand), int(seed), int(share), int(fee), int(decay), float(rho),
                    float(kappa), back)
orchard, season = settings.orchard(), settings.season()
a = _analysis(orchard)
s = a.stats
if a.problems:
    st.error("Der Prüfer fand Regelverletzungen in der Rechnung: " + "; ".join(a.problems))

# --- Akt 1: Ergebnis ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Akt 1: Ergebnis")
st.caption(f"{a.n_paths} Marktpfade mit je 90 Jahren (die ersten 22 Jahre zählen nicht mit), alle Regeln auf denselben Pfaden. "
           f"Streuung des Preisindex (ln p) im Mittel: {fmt(a.spread, 2)}. Jahresüberschuss gleichmäßig: {eur(s['gleich'].profit)}.")
m1, m2, m3 = st.columns(3)
# Delta immer als „mein Wert minus Referenz“ (Referenz: gleichmäßig pflanzen); mehr Überschuss ist besser, also delta_color="normal".
m1.metric("Dem Preis folgen", eur(s["folgend"].profit), delta=f"{s['folgend'].delta_pct:+.1f} % gegen gleichmäßig".replace(".", ","), delta_color="normal")
m2.metric("Vorausschau (öffentliche Pflanzzahlen)", eur(s["voraus"].profit), delta=f"{s['voraus'].delta_pct:+.1f} % gegen gleichmäßig".replace(".", ","),
          delta_color="normal")
m3.metric("Entgangene Verkäufe (gleichmäßig)", f"{fmt(s['gleich'].lost, 0)} Bäume/Jahr", delta=f"{fmt(s['gleich'].waste, 0)} verworfen",
          delta_color="off")
st.plotly_chart(build_timeline(a), width="stretch", key="chart_timeline")
st.caption("Ein Marktpfad (Pfad 1 von " + str(a.n_paths) + "): gepflanzte Bäume je Jahr nach Regel (links) und Preisindex des Marktes (rechts, gepunktet). "
           "Wer dem Preis folgt, pflanzt im Takt des Marktes; gleichmäßig ändert sich nichts.")
st.plotly_chart(build_rule_bars(a), width="stretch", key="chart_rules")
st.caption("Jahresüberschuss je Regel gegen gleichmäßiges Pflanzen, Mittel über alle Pfade; Fehlerbalken: zwei Standardfehler der gepaarten Differenz.")

kind_core, text_core = verdict_core(a)
season_a = _season(season) if st.session_state.get("season_done") == season else None

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a, text_core, season_a, verdict_season(season_a)[1] if season_a else ""))
if "pdf_bytes" in st.session_state and st.session_state["pdf_bytes"][0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf_bytes"][1], file_name="pflanzzyklus.pdf", mime="application/pdf",
                       key="pdf_download")

# --- Kernfrage ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt es sich, dem Preis zu folgen?")
getattr(st, kind_core)(text_core)

rows_table = []
for r in C.RULES:
    x = s[r]
    rows_table.append({
        "Regel": C.RULE_NAME[r], "Überschuss [EUR/Jahr]": round(x.profit), "gegen gleichmäßig [%]": round(x.delta_pct, 1),
        "Standardfehler [Prozentpunkte]": round(x.delta_se_pct, 2), "entgangene Verkäufe [Bäume/Jahr]": round(x.lost),
        "verworfen [Bäume/Jahr]": round(x.waste), "gepflanzt [Bäume/Jahr]": round(x.planted),
    })
st.dataframe(rows_table, hide_index=True, width="stretch")
st.caption("Referenz „Preis in 9 Jahren bekannt“ ist keine Obergrenze, sondern eine Regel mit Wissen: die Obergrenze im Erwartungswert zeigt der Tab 🧮 Exakt.")

# --- Experiment ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: ab welcher Marktreaktion lohnt es sich nicht mehr, dem Preis zu folgen?")
st.markdown(
    f"Derselbe Hof ({settings.dry} % Dürrejahre, Verkaufsfenster {settings.window} Jahre, Absatz {settings.demand}), die Marktreaktion von "
    f"{fmt(C.PHI_SWEEP[0])} bis {fmt(C.PHI_SWEEP[-1])}, jede Zelle über **{C.N_PATHS_SWEEP} Pfade aus {len(C.SWEEP_SEEDS)} festen Pfadmengen**. "
    "Das rechnet einige Sekunden und läuft erst auf Klick."
)
if st.button("Experiment rechnen", key="sweep_start"):
    st.session_state["sweep_done"] = orchard
if st.session_state.get("sweep_done") == orchard:
    with st.spinner("Rechne alle Regeln über alle Marktreaktionen ..."):
        rows = _sweep(orchard)
    kind_s, text_s = verdict_sweep(rows)
    getattr(st, kind_s)(text_s)
    st.plotly_chart(build_sweep(rows), width="stretch", key="chart_sweep")
    st.caption("Preisstreuung (ln p) je Marktreaktion: " + ", ".join(f"{fmt(r['phi'])}: {fmt(r['spread'], 2)}" for r in rows) + ".")

st.markdown("---")

# --- Akt 2 -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎄 Akt 2: Hebel-Check der Saison")
st.markdown(
    f"2 000 Bäume, 27 Verkaufstage (27.11. bis 23.12.), Hofverkauf täglich und Wochenmarkt mittwochs und samstags. Fünf Hebel auf denselben "
    f"Saisons: der **Markt** als zweiter Absatzweg, die **Marktmenge** nach Schattenpreis, der **Preis** (fest oder nach Tag und Restbestand), "
    f"und der **Preis der Unkenntnis** (Vorabkenntnis der Nachfrage). Die Sidebar steuert Marktanteil ({share} %), Kosten je Markttag ({fee} EUR), "
    f"Wertverlust ({decay} % je Tag) und ob der Hof Rückläufer zurücknimmt ({back}). Das rechnet einige Sekunden und läuft erst auf Klick."
)
if st.button("Hebel-Check rechnen", key="season_start"):
    st.session_state["season_done"] = season
if st.session_state.get("season_done") == season:
    with st.spinner("Rechne Marktregeln, Preise und die Schranke bei Vorabkenntnis ..."):
        sa = _season(season)
    kind_a2, text_a2 = verdict_season(sa)
    getattr(st, kind_a2)(text_a2)
    st.plotly_chart(build_levers(sa), width="stretch", key="chart_levers")
    st.caption(f"Basis: Gewinn ohne Markt bei Richtpreis {eur(sa.base)} je Saison (Mittel über {C.SEASON_N_TEST} Saisons, gepaarte Differenzen). "
               f"Feste Marktmenge: {sa.load_best} Bäume je Markttag (auf getrennten Saisons abgestimmt), bester fester Preis: {fmt(sa.m_best, 2)} mal Richtpreis.")
    for note in season_notes(sa):
        st.caption("• " + note)

st.markdown("---")

# --- Methodenvergleich ---------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    tab_rules, tab_exact, tab_market = st.tabs(["Die sieben Regeln", "🧮 Exakt (HiGHS)", "Das Marktmodell"])
    with tab_rules:
        st.markdown("Jede Regel legt nur die Pflanzmenge des Jahres fest. Alles andere (Ausfall, Wachstum, Verkauf) ist für alle gleich.")
        for r in C.RULES:
            st.markdown(f"**{C.RULE_NAME[r]}.** {C.RULE_TEXT[r]}")
        st.caption("Verkauft wird immer zuerst der älteste Baum im Wartebestand; Bäume, die das Verkaufsfenster überschreiten, werden verworfen.")
    with tab_exact:
        st.markdown(
            "Die **exakte Referenz** beantwortet: Was wäre im **Erwartungswert** möglich, wenn der ganze Preispfad im Voraus bekannt wäre? "
            f"Ein lineares Programm (HiGHS) wählt über {C.EXACT_HORIZON} Pflanzjahre die Pflanzmengen und die Verkäufe je Jahr und Alter; Ausfall im "
            "Erwartungswert, Nachfrage fest. Die Regeln pflanzen im selben Modell und verkaufen mit dem optimalen Verkaufsplan. "
            f"Gemittelt über {C.EXACT_PATHS} Marktpfade; es ist **keine Obergrenze für den zufälligen Betrieb**."
        )
        if st.button("Exakte Referenz berechnen", key="exact_start"):
            st.session_state["exact_done"] = orchard
        if st.session_state.get("exact_done") == orchard:
            with st.spinner("Löse die Pflanzpläne als LP ..."):
                ex = _exact(orchard)
            kind_e, text_e = verdict_exact(ex)
            getattr(st, kind_e)(text_e)
            st.plotly_chart(build_exact(ex), width="stretch", key="chart_exact")
            st.caption("Pfad 1: Pflanzmenge je Jahr im Optimum gegen gleichmäßig und dem Preis folgen; gepunktet der Preisindex der Verkaufsjahre.")
            st.dataframe([{"Regel": C.RULE_NAME[r], "Abstand zum Optimum [%]": round(v, 1)} for r, v in ex.gap_mean.items()],
                         hide_index=True, width="stretch")
    with tab_market:
        st.markdown(
            f"Der Marktpreis folgt dem **Angebot aus Pflanzungen von vor 7 bis 12 Jahren**: ln p = −φ · ln S + Nachfragestoß, S = Σ w_k · Q(t−k) · Angebotsstoß, "
            f"und die Betriebe pflanzen im Markt so viel, wie der heutige Preis nahelegt (Q = p hoch η, begrenzt auf {C.Q_CAP[0]} bis {C.Q_CAP[1]}). "
            f"Der Regler „Marktreaktion“ ist das Produkt φ · η. Unter etwa 1,1 klingt eine Störung ab (lineare Rechnung: Schwelle 1,09), darüber schwingt der Markt, "
            "bis die Begrenzung greift. Der Preisindex wird je Pfad auf das Mittel 1 gebracht. Der Hof ist Preisnehmer: sein Verhalten wirkt nicht auf den Markt."
        )

# --- Mathematische Formulierung --------------------------------------------------------------------------------------------------------

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Jahrgänge.** Im Jahr $y$ werden $P_y$ Bäume gepflanzt; davon sind $N_y = P_y \, s_y \, q$ verkaufsfähig ($s_y$ Überleben, im Dürrejahr
$s_y = s_0 (1 - u)$ mit $u \sim U(0{,}2;\,0{,}6)$, $q = 0{,}8$ Verkaufsware). Ein Baum erreicht die Verkaufsgröße nach $\tau \in \{7,\dots,12\}$ Jahren
(Wahrscheinlichkeiten $0{,}05 / 0{,}20 / 0{,}30 / 0{,}25 / 0{,}15 / 0{,}05$), der ganze Jahrgang zusätzlich um $-1, 0, +1$ Jahre verschoben. Die Ankünfte
im Jahr $j$ sind $a_j = \sum_i N_i \, \pi_{j-i}$ mit dem Ankunftsprofil $\pi$.

**Verkauf.** Der Wartebestand $x_{j,k}$ ($k$ Jahre seit Erreichen der Größe, $k < W$ mit Verkaufsfenster $W$) wird zuerst aus den ältesten Bäumen
bedient: $s_j = \min(D_j, \sum_k x_{j,k})$, Nachfrage $D_j = \bar D \, \varepsilon_j$ (lognormal, Streuung 0,2). Bäume im Alter $W$ werden verworfen.

**Überschuss.** $\text{cash}_j = \text{Preis} \cdot p_j \cdot s_j - c_{\text{plant}} P_j - c_{\text{care}} \,(\text{noch nicht angekommene Bäume} + \text{Wartebestand})$ mit
$c_{\text{plant}} = 1{,}60$ EUR, $c_{\text{care}} = 1{,}50$ EUR je Baum und Jahr, Preis 45 EUR je Baum bei Preisindex $p_j = 1$.

**Regeln.** Sollmenge $P^\ast = \bar D / (s \cdot q)$ mit mittlerem Überleben $s$. Folgen: $P_y = P^\ast p_y^{\gamma}$; antizyklisch: $P^\ast p_y^{-\gamma}$; Vorausschau:
$P^\ast \hat p_{y+9}^{\gamma}$ mit $\hat p_{y+9} = \big(\sum_k w_k Q_{y+9-k}\big)^{-\varphi}$ aus den öffentlichen Pflanzmengen; Nachpflanzen: $P^\ast + (P^\ast s q - N_{y-1})^+ / (s q)$.
Alle begrenzt auf $[0{,}3;\,2]\cdot P^\ast$ und die Fläche ($1{,}5 \, P^\ast$).

**Exakte Referenz (LP).** Variablen $P_y \in [0, P_{\max}]$, Bestände $x_{j,k} \ge 0$ und Verkäufe $y_{j,k} \le x_{j,k}$ mit
$x_{j,0} = a_j$, $x_{j,k} = x_{j-1,k-1} - y_{j-1,k-1}$, $\sum_k y_{j,k} \le \bar D$. Maximiere
$\sum_j \big(\text{Preis}\, p_j \sum_k y_{j,k} - c_{\text{plant}} P_j - c_{\text{care}} (u_j + \sum_k (x_{j,k} - y_{j,k}))\big)$; $u_j$ sind die noch nicht angekommenen Bäume, linear in $P$.

**Akt 2.** Hof: $D^H_t \sim \text{Poisson}(\lambda^H_t U_t (1 - \kappa (m - 1))^+)$, Markt entsprechend mit $\lambda^M_t$ an Mittwoch und Samstag. Schattenpreis eines Baums $\Delta V_t(s) = V_t(s) - V_t(s-1)$ aus der
Rückwärtsrechnung $V_t(s) = \max_m \mathbb{E}\big[(p_H m - c)\min(D, s) + V_{t+1}(s - \min(D, s))\big]$; geladen wird, solange
$q_l (p_M m - f \Delta V) \ge (1 - f)\Delta V + c_{\text{carry}}$ mit $q_l = \Pr(D^M \ge l)$ und $f$ dem Wertfaktor eines Rückläufers. Schranke: Aufzählung der Markttage,
Mengen frei verteilbar.
"""
    )

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
- **Angenommener Markt.** Das Spinnweb-Modell ist ein Lehrbuchmodell; wie stark der Zyklus am deutschen Markt wirklich ist, ist nicht belegt.
  Der Regler zeigt, was passiert, wenn er so stark ist, nicht, dass er es ist.
- **Fester Absatz.** Der Hof setzt rund 2 000 Bäume im Jahr ab, unabhängig vom Preis. Mit Händlerabsatz in beliebiger Menge fiele die Rechnung anders aus.
- **Ein Preis je Baum.** Größenklassen und der Erntezeitpunkt (wann ein Baum zu groß wird, um ihn zu verkaufen) sind nicht modelliert.
- **Richtwerte, keine Messung.** Pflanzdichte, Preis je Meter und Ausfallspanne stammen aus Verbandsangaben, alles andere sind Annahmen
  (siehe README). Keine Empfehlung für einen Betrieb.
- **Akt 2 ist ein Check, keine Planung.** Die Hebel der Saison hängen an Annahmen über Marktanteil, Standgebühr, Wertverlust und die Rücknahme der Rückläufer.
"""
)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
