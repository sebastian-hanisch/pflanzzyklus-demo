"""PDF-Export: erzeugt gültige Bytes mit und ohne Akt 2, auch mit Sonderzeichen in den Texten."""

import pfz_evaluation as ev
from pfz_pdf_export import build_pdf


def test_pdf_without_the_season(default_analysis):
    data = build_pdf(default_analysis, ev.verdict_core(default_analysis)[1])
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_pdf_with_the_season_and_special_characters(default_analysis):
    sa = ev.analyse_season(ev.SeasonSettings())
    data = build_pdf(default_analysis, "Text mit „Anführungszeichen“, Gedankenstrich – und Euro € sowie Umlauten äöüß.", sa, ev.verdict_season(sa)[1])
    assert data[:5] == b"%PDF-" and len(data) > 2000
