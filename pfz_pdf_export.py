"""PDF-Export des aktuellen Vergleichs (fpdf2). Nur Kernschriften: kein Gedankenstrich, kein Euro-Zeichen, Umlaute sind erlaubt."""

from fpdf import FPDF

import pfz_constants as C


def _txt(s: str) -> str:
    """Kernschriften kennen kein Unicode-Sonderzeichen außerhalb von Latin-1; Markdown-Sterne entfernen."""
    return s.replace("**", "").replace("–", "-").replace("—", "-").replace("€", "EUR").replace("→", "->").replace("„", '"').replace("“", '"')


def _para(pdf, height, text):
    """Absatz über die volle Breite; der Cursor geht danach an den linken Rand (fpdf2 lässt ihn sonst rechts stehen)."""
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


def build_pdf(a, verdict_text: str, season=None, season_text: str = "") -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    _para(pdf, 8, _txt("Pflanzzyklus: Pflanzregeln im Vergleich"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    _para(pdf, 5, _txt(
        f"Szenario: Marktreaktion {s.phi:.2f}, Stärke der Regeln {s.gamma:.2f}, {s.dry} % Dürrejahre, Verkaufsfenster {s.window} Jahre, "
        f"Absatz {s.demand} Bäume je Jahr, Seed {s.seed}, {a.n_paths} Marktpfade. Streuung des Preises (ln p): {a.spread:.2f}."))
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    _para(pdf, 5, "Kernfrage")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5, _txt(verdict_text))
    pdf.ln(3)
    widths = [66, 28, 28, 22, 22]
    heads = ["Regel", "Überschuss/Jahr EUR", "gegen gleichm. %", "entg. Verk.", "verworfen"]
    pdf.set_font("Helvetica", "B", 8)
    for w, text in zip(widths, heads):
        pdf.cell(w, 6, _txt(text), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in C.RULES:
        st = a.stats[r]
        cells = [C.RULE_NAME[r], f"{st.profit:,.0f}".replace(",", "."), f"{st.delta_pct:+.1f}".replace(".", ","), f"{st.lost:.0f}", f"{st.waste:.0f}"]
        for w, text in zip(widths, cells):
            pdf.cell(w, 6, _txt(text), border=1)
        pdf.ln()
    if season is not None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        _para(pdf, 5, "Akt 2: Saison")
        pdf.set_font("Helvetica", "", 9)
        _para(pdf, 5, _txt(season_text))
        for lv in season.levers:
            _para(pdf, 5, _txt(f"- {lv.name}: {lv.value:,.0f} EUR je Saison ({lv.pct:+.1f} %)".replace(",", ".")))
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4, _txt("Synthetischer Hof, Marktmodell angenommen; alle Eingangszahlen sind Annahmen oder Verbandsrichtwerte, keine Messung aus "
                       "einem echten Betrieb. Quelle: sebastianhanisch.net"))
    return bytes(pdf.output())
