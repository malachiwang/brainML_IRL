"""Experiment 5: K0 variants are defined as documented; simulation and recovery are exact and leak-free."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from reward_pairs import ck_init_mechanisms as m
from reward_pairs.authors_models import LIMITS, AuthorsTrials, FREE, evaluate as authors_evaluate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/02_comp_mod_RP_task_data_in.RData'
RESULTS = ROOT/'docs/experiment5_ck_init_mechanisms.json'
VALUES = np.array([1, 2, 2, 3, 3, 4, 4, 5.])
TRAINED = np.array([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
V = np.array([.1, 2.4, 1.3, 3.9, .7, 1.8, 2.9, .4])
THETA = [.15, 1.8, .4, 1.2]


def _trials(seed, n):
    rng = np.random.default_rng(seed)
    pairs = TRAINED[rng.integers(0, 8, n)]
    pairs[::2] = pairs[::2, ::-1]
    better = VALUES[pairs[:, 0]] > VALUES[pairs[:, 1]]
    return AuthorsTrials(pairs, np.where(rng.random(n) < .75, better, ~better), VALUES[pairs])


@pytest.fixture
def fold():
    return SimpleNamespace(fit_trials=_trials(1, 160), validation_trials=_trials(2, 50), fit_sessions=np.repeat([1, 2, 3, 4], 40))


def test_variant_definitions():
    d = V-V.mean()
    k = {v: m.initial_kernel(v, V, 5) for v in m.VARIANTS}
    np.testing.assert_array_equal(k['released'], V)
    np.testing.assert_allclose(k['level_shift']-k['level_shift'].mean(), d, atol=1e-15)  # same pattern and spread
    assert k['level_shift'].mean() == pytest.approx(.5)
    np.testing.assert_allclose(k['shrink_0.5'], .5+.5*d); np.testing.assert_allclose(k['shrink_0.25'], .5+.25*d)
    assert np.all(k['uniform_0.5'] == .5) and np.allclose(k['uniform_mean'], V.mean())
    assert sorted(k['permuted']) == sorted(V) and not np.any(k['permuted'] == V)  # derangement of the same values
    np.testing.assert_array_equal(k['permuted'], m.initial_kernel('permuted', V, 5))  # deterministic
    assert k['minmax'].min() == 0 and k['minmax'].max() == 1
    np.testing.assert_array_equal(np.argsort(k['minmax']), np.argsort(V))  # rank preserving
    with pytest.raises(ValueError):
        m.initial_kernel('bogus', V, 1)
    with pytest.raises(ValueError):
        m.initial_kernel('minmax', np.ones(8), 1)


def test_uniform_level_irrelevant_when_appearance_counts_match():
    """Analytical claim in the module docstring: a uniform K0 level enters only through n_L != n_R."""
    pairs = np.tile([[0, 1], [1, 0]], (20, 1))  # both stimuli always shown together: equal counts
    t = AuthorsTrials(pairs, np.random.default_rng(0).random(40) < .6, VALUES[pairs])
    a = authors_evaluate(t, 19, THETA, V, np.zeros(8)).probability_left
    b = authors_evaluate(t, 19, THETA, V, np.full(8, .5)).probability_left
    np.testing.assert_allclose(a, b, atol=1e-13)
    unequal = _trials(3, 80)
    a = authors_evaluate(unequal, 19, THETA, V, np.zeros(8)).probability_left
    b = authors_evaluate(unequal, 19, THETA, V, np.full(8, .5)).probability_left
    assert not np.allclose(a, b)


def test_fit_real_keeps_q0_and_bounds(fold):
    saved = dict(theta=THETA, optimization=dict(optimizer_objective_deviance=0.), validation=dict(nll=0.))
    row = m.fit_real(('synthetic', 4, fold, V.copy(), V.copy(), saved, 3, False))
    assert row['q0'] == V.tolist()
    for v in m.VARIANTS:
        x = row['variants'][v]
        assert x['status'] == 'ok' and x['carried_ok']
        assert x['k0'] == m.initial_kernel(v, V, 4).tolist()
        th = np.array(x['theta']); assert np.all(th >= LIMITS[:4, 0]) and np.all(th <= LIMITS[:4, 1])
        assert sum(x['session_nll'].values()) == pytest.approx(x['train_nll'], abs=1e-9)
        assert authors_evaluate(fold.fit_trials, 19, th, V, np.array(x['k0'])).deviance == pytest.approx(x['deviance'], abs=1e-7)


def test_simulation_ignores_observed_choices_and_is_reproducible(fold):
    flipped = SimpleNamespace(fit_trials=AuthorsTrials(fold.fit_trials.pairs, ~fold.fit_trials.left, fold.fit_trials.rewards),
                              validation_trials=AuthorsTrials(fold.validation_trials.pairs, ~fold.validation_trials.left,
                                                              fold.validation_trials.rewards), fit_sessions=fold.fit_sessions)
    a = m.simulate(fold, THETA, V, np.full(8, .5), 11)
    b = m.simulate(flipped, THETA, V, np.full(8, .5), 11)
    c = m.simulate(fold, THETA, V, np.full(8, .5), 12)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.left, y.left); np.testing.assert_array_equal(x.pairs, y.pairs)
    assert not np.array_equal(a[0].left, c[0].left)
    # S5 simulation continues from the simulated S1-S4 end state
    tr = authors_evaluate(a[0], 19, THETA, V, np.full(8, .5))
    va = authors_evaluate(a[1], 19, THETA, tr.q[-1], tr.h[-1])
    assert np.all(np.isfinite(va.log_loss))


def test_recovery_records_optimization_and_identifiability(fold):
    ds = m.synthetic_dataset(fold, V, V, 4, 'uniform_0.5', THETA, 7, fit_inits=('uniform_0.5', 'released'), n_starts=4)
    for f, r in ds['fits'].items():
        assert r['nll_at_truth_minus_fit'] >= -1e-6  # the reported optimum is never worse than the truth
        assert r['deviance'] <= min(r['multistart_deviance'], r['truth_start_deviance'] or np.inf) + 1e-12
        assert r['delta_nll_true_alpha'] >= -1e-6
    assert ds['fits']['uniform_0.5']['s5_mean_abs_dp_vs_matched'] == 0
    again = m.synthetic_dataset(fold, V, V, 4, 'uniform_0.5', THETA, 7, fit_inits=('uniform_0.5', 'released'), n_starts=4)
    assert json.dumps(ds, sort_keys=True) == json.dumps(again, sort_keys=True)


class _Guarded(dict):
    def __getitem__(self, key):
        if key == 'data_4_analysis_RP_test':
            raise AssertionError('final-test table accessed')
        return super().__getitem__(key)


@pytest.mark.skipif(not RAW.exists() or shutil.which('Rscript') is None, reason='Raw .RData or Rscript unavailable.')
def test_released_variant_reproduces_saved_fit_without_final_test(monkeypatch):
    spec = importlib.util.spec_from_file_location('run_e5', ROOT/'scripts/run_ck_init_mechanisms.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    real = mod.load_rdata(RAW)
    monkeypatch.setattr(mod, 'load_rdata', lambda path: _Guarded(real))
    folds = json.loads((ROOT/'docs/history_diagnostics_results.json').read_text())
    p = next(r['participant'] for r in folds['rows'] if r['session'] == 5)
    ((_, index, fold, q0, v, saved),) = mod.load(RAW, folds, [p])
    from reward_pairs.authors_audit import random_starts
    f = m._fit(fold, q0, m.initial_kernel('released', v, index), random_starts(index, 200)[:, list(FREE[19])])
    assert abs(f['deviance']-saved['optimization']['optimizer_objective_deviance']) < 1e-6
    assert np.array_equal(q0, v)


@pytest.mark.skipif(shutil.which('git') is None, reason='git unavailable')
def test_authors_modules_unchanged():
    out = subprocess.run(['git', 'status', '--porcelain', '--', 'src/reward_pairs/'], cwd=ROOT, capture_output=True, text=True, check=True)
    assert not [l for l in out.stdout.splitlines() if '/authors_' in l]


@pytest.mark.skipif(not RESULTS.exists(), reason='Experiment 5 results not generated.')
def test_primary_results_cover_all_participants():
    r = json.loads(RESULTS.read_text())
    assert r['real']['participants'] == r['real']['analysed'] == 213 and not r['real']['excluded']
    assert r['configuration']['participant'] is None
    assert all(rec['n'] == 213*r['configuration']['seeds'] for rec in r['recovery'].values())
