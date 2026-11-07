"""Experiment 4: kernel initialization conditions change only K0 and are scored exactly and causally."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from reward_pairs import ck_initialization as ci
from reward_pairs.authors_models import LIMITS, AuthorsTrials, evaluate as authors_evaluate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/02_comp_mod_RP_task_data_in.RData'
RESULTS = ROOT/'docs/experiment4_ck_initialization.json'
VALUES = np.array([1, 2, 2, 3, 3, 4, 4, 5.])
TRAINED = np.array([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
THETA = [.3, 1.5, .4, 1.1]


def _trials(seed, n):
    rng = np.random.default_rng(seed)
    pairs = TRAINED[rng.integers(0, 8, n)]
    pairs[::2] = pairs[::2, ::-1]
    better = VALUES[pairs[:, 0]] > VALUES[pairs[:, 1]]
    return AuthorsTrials(pairs, np.where(rng.random(n) < .75, better, ~better), VALUES[pairs])


@pytest.fixture
def fold():
    return SimpleNamespace(fit_trials=_trials(1, 160), validation_trials=_trials(2, 50), fit_sessions=np.repeat([1, 2, 3, 4], 40))


Q0 = np.linspace(.2, 4.1, 8)
SAVED = dict(theta=THETA, optimization=dict(optimizer_objective_deviance=0.), initial_validation_q=[0.]*8,
             initial_validation_h=[0.]*8, validation=dict(nll=0.))


def _job(fold, n=4, profile=False):
    return ('synthetic', 3, fold, Q0.copy(), Q0.copy(), SAVED, n, False, profile)


def test_conditions_change_only_k0():
    for c in ci.CONDITIONS:
        k0 = ci.initial_kernel(c, Q0)
        assert k0.shape == (8,)
    np.testing.assert_array_equal(ci.initial_kernel('C0', Q0), Q0)
    assert ci.initial_kernel('C0', Q0) is not Q0  # a copy, never an alias of Q0
    np.testing.assert_array_equal(ci.initial_kernel('C1', Q0), np.full(8, .5))
    np.testing.assert_array_equal(ci.initial_kernel('C2', Q0), np.zeros(8))
    with pytest.raises(ValueError):
        ci.initial_kernel('C3', Q0)


def test_fit_keeps_q0_identical_bounds_and_is_reproducible(fold):
    a = ci.fit_participant(_job(fold))
    b = ci.fit_participant(_job(fold))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a['checks']['q0_identical'] and a['q0'] == Q0.tolist()
    assert a['k0']['C1'] == [.5]*8 and a['k0']['C2'] == [0.]*8 and a['k0']['C0'] == Q0.tolist()
    for c in ci.CONDITIONS:
        x = a['conditions'][c]
        assert x['status'] == 'ok' and x['carried_ok']
        th = np.array(x['theta'])
        assert np.all(th >= LIMITS[:4, 0]) and np.all(th <= LIMITS[:4, 1])
        # replay with this condition's K0 reproduces the optimizer objective
        assert authors_evaluate(fold.fit_trials, 19, th, Q0, np.array(a['k0'][c])).deviance == pytest.approx(x['deviance'], abs=1e-7)


def test_kernel_recurrence_limits_and_unequal_counts(fold):
    t = fold.fit_trials
    for alpha in (0., .37, 1.):
        theta = [alpha, 1.5, .4, 1.1]
        out = authors_evaluate(t, 19, theta, Q0, np.full(8, .5))
        k = np.full(8, .5)
        for i, (l, r) in enumerate(t.pairs):
            np.testing.assert_allclose(out.h[i], k, rtol=0, atol=1e-15)
            k = k.copy(); k[l] += alpha*(float(t.left[i])-k[l]); k[r] += alpha*(float(not t.left[i])-k[r])
        if alpha == 0:
            assert np.all(out.h == .5)
        if alpha == 1:
            last = {}
            for i, (l, r) in enumerate(t.pairs):
                last[l], last[r] = float(t.left[i]), float(not t.left[i])
            assert all(out.h[-1, s] == v for s, v in last.items())
    # Uniform K0 of 0 and 0.5 agree on the first prediction but differ once appearance counts differ.
    p0 = authors_evaluate(t, 19, THETA, Q0, np.zeros(8)).probability_left
    p5 = authors_evaluate(t, 19, THETA, Q0, np.full(8, .5)).probability_left
    assert p0[0] == pytest.approx(p5[0], abs=1e-15)
    assert not np.allclose(p0, p5)


def test_session_losses_sum_to_total_and_s5_uses_final_states(fold):
    s, (tr, va) = ci.score(fold, THETA, Q0, np.full(8, .5))
    assert sum(s['session_nll'].values()) == pytest.approx(s['train_nll'], abs=1e-9)
    n = np.bincount(fold.fit_sessions)[1:]
    assert np.dot(n, [s['session_loss'][k] for k in '1234'])/n.sum() == pytest.approx(s['train_loss'], abs=1e-12)
    np.testing.assert_array_equal(va.q[0], tr.q[-1]); np.testing.assert_array_equal(va.h[0], tr.h[-1])
    assert s['carried_ok']
    labels = ci.segment_labels(fold.fit_sessions)
    assert (labels == 'S1 early').sum() == 14 and (labels == 'S1 late').sum() == 26


def test_session5_probabilities_precede_choice_updates(fold):
    _, (_, base) = ci.score(fold, THETA, Q0, Q0)
    t = 17
    left = fold.validation_trials.left.copy(); left[t] = ~left[t]
    alt = SimpleNamespace(fit_trials=fold.fit_trials, fit_sessions=fold.fit_sessions,
                          validation_trials=AuthorsTrials(fold.validation_trials.pairs, left, fold.validation_trials.rewards))
    _, (_, changed) = ci.score(alt, THETA, Q0, Q0)
    np.testing.assert_array_equal(base.probability_left[:t+1], changed.probability_left[:t+1])
    assert not np.array_equal(base.probability_left[t+1:], changed.probability_left[t+1:])


def test_failed_conditions_are_excluded_not_hidden(fold):
    good = ci.fit_participant(_job(fold))
    bad = json.loads(json.dumps(good)); bad['participant'] = 'broken'
    bad['conditions']['C1'] = dict(status='failed', error='forced')
    kept, excluded = ci.usable([good, bad])
    assert [r['participant'] for r in kept] == ['synthetic'] and excluded == ['broken']


@pytest.mark.skipif(shutil.which('git') is None, reason='git unavailable')
def test_benchmark_modules_unmodified():
    out = subprocess.run(['git', 'status', '--porcelain', '--', 'src/reward_pairs/authors_models.py',
                          'src/reward_pairs/authors_fitting.py', 'src/reward_pairs/authors_audit.py',
                          'src/reward_pairs/authors_evaluation.py'], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ''


class _Guarded(dict):
    def __getitem__(self, key):
        if key == 'data_4_analysis_RP_test':
            raise AssertionError('final-test table accessed')
        return super().__getitem__(key)


@pytest.mark.skipif(not RAW.exists() or shutil.which('Rscript') is None, reason='Raw .RData or Rscript unavailable.')
def test_c0_reproduces_saved_benchmark_without_final_test(monkeypatch):
    spec = importlib.util.spec_from_file_location('run_ck_initialization', ROOT/'scripts/run_ck_initialization.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    real = mod.load_rdata(RAW)
    monkeypatch.setattr(mod, 'load_rdata', lambda path: _Guarded(real))
    folds = json.loads((ROOT/'docs/history_diagnostics_results.json').read_text())
    p = next(r['participant'] for r in folds['rows'] if r['session'] == 5)
    (job,) = mod.build_jobs(RAW, folds, [p], 200, True, False)
    row = ci.fit_participant(job)
    for k, tol in mod.TOLERANCE.items():
        assert row['checks'][k] <= tol, k
    assert row['checks']['q0_identical']


@pytest.mark.skipif(not RESULTS.exists(), reason='Experiment 4 results not generated.')
def test_all_213_participants_in_primary_results():
    r = json.loads(RESULTS.read_text())
    assert r['summary']['participants'] == r['summary']['analysed'] == 213 and not r['summary']['excluded']
    assert r['configuration']['participant'] is None
