"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import pfz_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "phi_slider": SettingSpec("phi", float, C.DEFAULT_PHI, C.PHI_MIN, C.PHI_MAX, 0.05),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX, 0.25),
    "dry_slider": SettingSpec("dry", int, C.DEFAULT_DRY, C.DRY_MIN, C.DRY_MAX, 5),
    "window_slider": SettingSpec("window", int, C.DEFAULT_WINDOW, C.WINDOW_MIN, C.WINDOW_MAX, 1),
    "demand_slider": SettingSpec("demand", int, C.DEFAULT_DEMAND, C.DEMAND_MIN, C.DEMAND_MAX, 250),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "share_slider": SettingSpec("share", int, C.DEFAULT_SHARE, C.SHARE_MIN, C.SHARE_MAX, 5),
    "fee_slider": SettingSpec("fee", int, C.DEFAULT_FEE, C.FEE_MIN, C.FEE_MAX, 20),
    "decay_slider": SettingSpec("decay", int, C.DEFAULT_DECAY, C.DECAY_MIN, C.DECAY_MAX, 1),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX, 0.1),
    "kappa_slider": SettingSpec("kappa", float, C.DEFAULT_KAPPA, C.KAPPA_MIN, C.KAPPA_MAX, 0.5),
    "back_radio": SettingSpec("back", str, "ja", choices=C.BACK_OPTIONS),
}
PRESET_KEYS = {"phi": "phi_slider", "gamma": "gamma_slider", "dry": "dry_slider", "window": "window_slider", "demand": "demand_slider",
               "seed": "seed_input", "share": "share_slider", "fee": "fee_slider", "decay": "decay_slider", "rho": "rho_slider",
               "kappa": "kappa_slider", "back": "back_radio"}


def snap(spec: SettingSpec, value):
    """Begrenzt auf die Reglergrenzen und rastet auf den Schritt ein (sonst lehnt der Regler den Wert ab)."""
    if spec.choices is not None:
        return value if value in spec.choices else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, round(value, 6))
        if spec.caster is int:
            value = int(round(value))
    return value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = snap(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
