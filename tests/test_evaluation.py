"""Auswertung und Verdikte: Schwellen und jede Zahl im Text an handgesetzten Werten; Invarianten der Mehr-Pfad-Rechnung."""

import numpy as np
import pytest

import pfz_constants as C
import pfz_evaluation as ev


def _analysis(delta_by_rule, se=0.2, spread=0.31):
    stats = {}
    for r in C.RULES:
        d = delta_by_rule.get(r, 0.0)
        stats[r] = ev.RuleStats(profit=50000 * (1 + d / 100), delta=500 * d, delta_pct=d, delta_se_pct=0.0 if r == "gleich" else se, lost=10.0,
                                waste=20.0, planted=3000.0)
    return ev.Analysis(ev.OrchardSettings(), None, spread, stats, {}, 300, [])


def test_verdict_core_warns_when_following_the_price_costs_money():
    kind, text = ev.verdict_core(_analysis({"folgend": -12.34}))
    assert kind == "warning" and "12,3 %" in text and "0,31" in text


def test_verdict_core_is_success_when_following_pays():
    kind, text = ev.verdict_core(_analysis({"folgend": 4.0}))
    assert kind == "success" and "4,0 %" in text


def test_verdict_core_is_info_inside_the_tolerance():
    kind, text = ev.verdict_core(_analysis({"folgend": -0.6}))
    assert kind == "info" and "fast gleichgültig" in text


def test_verdict_core_tolerance_scales_with_the_standard_error():
    """-1,5 % bei Standardfehler 1,0 liegt innerhalb von zwei Standardfehlern: kein Warnsignal."""
    assert ev.verdict_core(_analysis({"folgend": -1.5}, se=1.0))[0] == "info"
    assert ev.verdict_core(_analysis({"folgend": -2.5}, se=1.0))[0] == "warning"


def test_verdict_core_names_a_better_alternative_only_when_it_clears_the_tolerance():
    kind, text = ev.verdict_core(_analysis({"folgend": -5.0, "lueck": 3.0}))
    assert "Beste Alternative: Lückenfüller mit +3,0 %" in text
    _, text2 = ev.verdict_core(_analysis({"folgend": -5.0, "lueck": 0.4}))
    assert "Beste Alternative" not in text2 and "beste: Lückenfüller" in text2


def test_verdict_core_ignores_the_reference_rule_when_picking_the_best_alternative():
    _, text = ev.verdict_core(_analysis({"folgend": -5.0, "perfekt": 9.0, "lueck": 0.1}))
    assert "Preis in 9 Jahren" not in text


def _rows(deltas):
    return [dict(phi=phi, spread=0.1 * phi, delta={r: (d if r == "folgend" else 0.0) for r in C.RULES},
                 se={r: 0.3 for r in C.RULES}, profit=1.0) for phi, d in zip(C.PHI_SWEEP, deltas)]


def test_verdict_sweep_names_the_first_phi_with_a_cost_above_two_percent():
    kind, text = ev.verdict_sweep(_rows([0, -0.5, -1.9, -2.5, -10, -30, -60]))
    assert kind == "warning" and "ab einer Marktreaktion von 1,1" in text and "(2,5 %)" in text and "sind es 60,0 %" in text


def test_verdict_sweep_without_a_hit_is_info():
    kind, text = ev.verdict_sweep(_rows([0] * len(C.PHI_SWEEP)))
    assert kind == "info" and "mehr als 2 %" in text


def test_verdict_sweep_needs_two_standard_errors():
    rows = _rows([0, 0, 0, -2.5, 0, 0, 0])
    rows[3]["se"]["folgend"] = 2.0          # -2,5 % bei Standardfehler 2: nicht belegt
    assert ev.verdict_sweep(rows)[0] == "info"


def test_rule_stats_pair_the_differences_and_gleich_has_zero_delta(default_analysis):
    s = default_analysis.stats
    assert s["gleich"].delta == 0.0 and s["gleich"].delta_pct == 0.0 and s["gleich"].delta_se_pct == 0.0
    for r in C.RULES:
        assert s[r].profit == pytest.approx(s["gleich"].profit + s[r].delta)
        assert s[r].delta_pct == pytest.approx(s[r].delta / s["gleich"].profit * 100)


def test_analyse_is_deterministic_and_the_checker_is_quiet(default_analysis):
    again = ev.analyse(ev.OrchardSettings())
    assert again.problems == [] and default_analysis.problems == []
    assert {r: s.profit for r, s in again.stats.items()} == {r: s.profit for r, s in default_analysis.stats.items()}


def test_the_seed_changes_the_paths():
    a, b = ev.analyse(ev.OrchardSettings(seed=1)), ev.analyse(ev.OrchardSettings(seed=2))
    assert a.stats["gleich"].profit != b.stats["gleich"].profit


def test_sweep_rows_follow_the_fixed_grid_and_the_spread_grows():
    rows = ev.sweep_phi(ev.OrchardSettings())
    assert [r["phi"] for r in rows] == list(C.PHI_SWEEP)
    assert all(x["spread"] < y["spread"] for x, y in zip(rows, rows[1:]))
    assert all(r["delta"]["gleich"] == 0.0 for r in rows)


def test_exact_comparison_gaps_are_nonnegative_and_the_optimum_beats_gleich():
    ex = ev.exact_comparison(ev.OrchardSettings(), n_paths=3)
    assert all(g >= -1e-6 for g in ex.gap_mean.values())
    assert ex.optimum_vs_gleich > 0 and ex.horizon == C.EXACT_HORIZON
    kind, text = ev.verdict_exact(ex)
    assert kind == "info" and "Obergrenze nur für diese Erwartungswert-Rechnung" in text and "kein Plan" in text and "Am nächsten am Optimum" in text


def _lever(key, value, se=1.0, base=60000.0):
    return ev.Lever(key, key, value, se, value / base * 100)


def _season(values, **kw):
    levers = [_lever(k, v) for k, v in values.items()]
    details = dict(qloss_nv=15.0, qloss_fest=60.0, attended_nv=7.6, sold_m_nv=645.0, left_nv=125.0, left_kein=689.0, price_dp_left=787.0, n_markt_days=8)
    return ev.SeasonAnalysis(ev.SeasonSettings(**kw), 60000.0, levers, 110, 1.05, details)


def test_verdict_season_reports_the_market_lever_and_calls_the_rest_solved_below_three_percent():
    kind, text = ev.verdict_season(_season(dict(markt=9000.0, regel=700.0, preis=300.0, preisplan=0.0, unkenntnis=900.0)))
    assert kind == "info" and "+15,0 %" in text and "35 %" in text and "fast gelöst" in text and "„unkenntnis“ mit +1,5 %" in text and "Alle übrigen Hebel sind kleiner" in text


def test_verdict_season_warns_when_another_lever_is_large():
    kind, text = ev.verdict_season(_season(dict(markt=9000.0, regel=3000.0, preis=300.0, preisplan=0.0, unkenntnis=900.0)))
    assert kind == "warning" and "„regel“ mit +5,0 %" in text


def test_season_notes_explain_a_dead_price_plan_and_the_return_setting():
    notes = ev.season_notes(_season(dict(markt=1.0, regel=1.0, preis=1.0, preisplan=0.0, unkenntnis=1.0)))
    assert any("nie knapp" in n and "787" in n for n in notes)
    assert any("zurück" in n and "15 EUR" in n for n in notes)
    notes2 = ev.season_notes(_season(dict(markt=1.0, regel=1.0, preis=1.0, preisplan=50.0, unkenntnis=1.0), back="nein"))
    assert not any("nie knapp" in n for n in notes2) and any("nicht zurück" in n for n in notes2)


def test_formatting_helpers_use_the_decimal_comma_and_thousands_dots():
    assert ev.fmt(1.26) == "1,3" and ev.fmt(1.26, 2) == "1,26"
    assert ev.pct(-6.66, sign=True) == "-6,7 %" and ev.pct(3.0, sign=True) == "+3,0 %"
    assert ev.eur(52926.4) == "52.926 EUR"
    assert np.isclose(float(ev.fmt(0.31, 2).replace(",", ".")), 0.31)


def test_settings_split_into_the_two_acts():
    s = ev.Settings(phi=1.3, share=10, fee=0)
    assert s.orchard() == ev.OrchardSettings(phi=1.3) and s.season() == ev.SeasonSettings(share=10, fee=0)
    assert ev.SeasonSettings(decay=15, back="nein").params().decay == 0.15 and ev.SeasonSettings(back="nein").params().back is False


def _exact(gaps):
    return ev.ExactResult(10, gaps, 3.8, 0.5, {}, C.EXACT_HORIZON)


def test_verdict_exact_names_the_nearest_rule_once():
    kind, text = ev.verdict_exact(_exact(dict(gleich=3.7, folgend=8.0, voraus=5.2, nach=3.7, lueck=4.0, perfekt=5.2)))
    assert "Am nächsten am Optimum liegt gleichmäßiges Pflanzen (3,7 % darunter), dem Preis zu folgen liegt 8,0 % darunter" in text
    assert text.count("gleichmäßig pflanzen liegt") == 0 and "+3,8 %" in text and "10 Pfade" in text


def test_verdict_exact_names_a_better_rule_when_there_is_one():
    _, text = ev.verdict_exact(_exact(dict(gleich=3.7, folgend=8.0, voraus=5.2, nach=3.7, lueck=3.6, perfekt=5.2)))
    assert "Am nächsten am Optimum liegt Lückenfüller (3,6 % darunter), gleichmäßig pflanzen liegt 3,7 % darunter, dem Preis zu folgen 8,0 %" in text


def test_the_reference_rule_is_never_named_as_the_nearest():
    _, text = ev.verdict_exact(_exact(dict(gleich=3.7, folgend=8.0, voraus=5.2, nach=3.7, lueck=4.0, perfekt=0.5)))
    assert "Preis in 9 Jahren" not in text
