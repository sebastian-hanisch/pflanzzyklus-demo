"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_radio_options_do_not_raise():
    at = _fresh_app()
    assert at.radio, "kein Radio gefunden"
    for option in at.radio[0].options:
        at = _fresh_app()
        at.radio[0].set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (option, [str(e) for e in at.exception])


def test_all_sliders_at_their_minima_and_at_their_maxima_together():
    for edge in ("min", "max"):
        at = _fresh_app()
        for slider in at.slider:
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_permalink_values_are_clipped_and_snapped():
    at = AppTest.from_file(APP_PATH)
    at.query_params["phi"] = "9"
    at.query_params["fee"] = "149"
    at.query_params["back"] = "vielleicht"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["phi_slider"] == 1.4 and at.session_state["fee_slider"] == 140 and at.session_state["back_radio"] == "ja"


def test_heavy_views_run_only_on_click_and_show_their_verdicts():
    at = _fresh_app()
    assert not any("Markt als zweiter" in str(m.value) for m in list(at.info) + list(at.warning))
    labels = {b.label: i for i, b in enumerate(at.button)}
    for label in ("Experiment rechnen", "Hebel-Check rechnen", "Exakte Referenz berechnen"):
        at.button[labels[label]].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])
        labels = {b.label: i for i, b in enumerate(at.button)}
    texts = " ".join(str(m.value) for m in list(at.info) + list(at.warning) + list(at.success))
    assert "Markt als zweiter Absatzweg" in texts and "Marktreaktion von" in texts and "Optimum" in texts
