"""Presets und Permalink: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Begrenzen und Einrasten."""

import pytest

import pfz_constants as C
import pfz_evaluation as ev
import pfz_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_on_the_slider_grid():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)
                assert P.snap(spec, value) == pytest.approx(value), (name, key)


def test_default_preset_equals_the_default_settings():
    d = C.PRESETS["Mittlere Schwankung (Voreinstellung)"]
    assert ev.Settings(**d) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_defaults_are_inside_the_bounds():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        if spec.choices is None:
            assert spec.lo <= spec.default <= spec.hi


def test_presets_differ_from_each_other():
    values = [tuple(sorted(v.items())) for v in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_snap_clips_and_rounds_to_the_step():
    phi = P.SETTING_SPECS["phi_slider"]
    assert P.snap(phi, 1.12) == pytest.approx(1.10) and P.snap(phi, 9.0) == C.PHI_MAX and P.snap(phi, 0.0) == C.PHI_MIN
    fee = P.SETTING_SPECS["fee_slider"]
    assert P.snap(fee, 149) == 140 and isinstance(P.snap(fee, 149), int) and P.snap(fee, -5) == 0


def test_snap_replaces_an_unknown_choice_with_the_default():
    back = P.SETTING_SPECS["back_radio"]
    assert P.snap(back, "vielleicht") == "ja" and P.snap(back, "nein") == "nein"


def test_slider_bounds_are_consistent_with_the_constants():
    assert P.bounds("phi_slider") == (C.PHI_MIN, C.PHI_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert C.PHI_SWEEP[0] >= C.PHI_MIN and C.PHI_SWEEP[-1] <= C.PHI_MAX and C.DEFAULT_PHI in C.PHI_SWEEP
