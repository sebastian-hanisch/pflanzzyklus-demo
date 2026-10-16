"""Gleiches Szenario, gleicher Seed, anderer Prozess (PYTHONHASHSEED 1 und 2): identische Ergebnisse (DEMO-PLAYBOOK §1d)."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = (
    "import json, pfz_evaluation as ev, pfz_season as S, pfz_constants as C\n"
    "a = ev.analyse(ev.OrchardSettings(seed=3))\n"
    "sp = ev.SeasonSettings().params()\n"
    "u = S.draw_factors(sp, 30, 4)\n"
    "p, _ = S.simulate_season(sp, u, 5, S.price_fixed(1.0), S.markt_fixed(60))\n"
    "print(json.dumps({'rules': {r: s.profit for r, s in a.stats.items()}, 'season': float(p.sum())}))\n"
)


def _run(hash_seed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=180)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_results_do_not_depend_on_the_hash_seed():
    assert _run("1") == _run("2")
