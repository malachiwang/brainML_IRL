"""Experiment 3: the fixed-alpha profile wrapper is exact, bounded, causal, reproducible, and fail-safe."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reward_pairs import ck_alpha_profile as cap
from reward_pairs.authors_models import AuthorsTrials, TrainingObjective, evaluate as authors_evaluate
from reward_pairs.history_diagnostics import make_fold, replay_fold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/02_comp_mod_RP_task_data_in.RData'
VALUES = np.array([1, 2, 2, 3, 3, 4, 4, 5.])
TRAINED = np.array([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])


def _trials(seed, n):
    rng = np.random.default_rng(seed)
    pairs = TRAINED[rng.integers(0, 8, n)]
    pairs[::2] = pairs[::2, ::-1]
    better = VALUES[pairs[:, 0]] > VALUES[pairs[:, 1]]
    left = np.where(rng.random(n) < .75, better, ~better)
    return AuthorsTrials(pairs, left, VALUES[pairs])


@pytest.fixture
def setup():
    fold = SimpleNamespace(fit_trials=_trials(1, 160), validation_trials=_trials(2, 50), fit_sessions=np.repeat([1, 2, 3, 4], 40))
    q0, h0 = np.linspace(.2, 3., 8), np.linspace(.2, 3., 8)
    problem = TrainingObjective([fold.fit_trials], cap.MODEL, q0, h0)
    return fold, q0, h0, problem


def _job(fold, q0, h0, theta=(.2, 1.5, .4, 1.2), grid=(0., .1, 1.), n_random=3):
    tr, va = replay_fold(fold, 'ck', list(theta), q0, h0)
    saved = dict(theta=list(theta), validation=dict(nll=float(va.log_loss.sum())),
                 optimization=dict(optimizer_objective_deviance=tr.deviance, boundary_flags={}))
    return ('synthetic', 7, fold, q0, h0, saved, grid, n_random)


def test_fixed_alpha_never_moves_and_nuisance_respects_bounds(setup):
    _, _, _, problem = setup
    seen = []

    def spy(theta):
        seen.append(theta[0]); return problem(theta)
    spy.model = problem.model
    alphas = (0., .037, 1.)
    points = cap.profile_curve(spy, alphas, .037, np.array([1., .5, 1.]), cap.random_nuisance_starts(1, 3))
    assert set(seen) == set(alphas)  # every objective call used exactly one of the fixed grid values
    for a, p in zip(alphas, points):
        assert p['status'] == 'ok' and p['theta'][0] == a and p['alpha_ck'] == a
        x = np.array(p['theta'][1:])
        assert np.all(x >= cap.NUISANCE_BOUNDS[:, 0]) and np.all(x <= cap.NUISANCE_BOUNDS[:, 1])
    assert points[1]['anchor'] and not points[0]['anchor']


def test_constrained_objective_equals_authors_objective(setup):
    fold, q0, h0, problem = setup
    x = np.array([1.3, .35, 2.2])
    for a in (0., .25, 1.):
        dev, grad = cap.FixedAlphaObjective(problem, a)(x)
        full_dev, full_grad = problem(np.r_[a, x])
        assert dev == full_dev and np.array_equal(grad, full_grad[1:])
        assert dev == pytest.approx(authors_evaluate(fold.fit_trials, 19, np.r_[a, x], q0, h0).deviance, abs=1e-9)
    with pytest.raises(ValueError):
        cap.FixedAlphaObjective(problem, 1.01)


def test_free_parameter_gradient_matches_finite_differences(setup):
    _, _, _, problem = setup
    for a in (0., .3, 1.):
        f = cap.FixedAlphaObjective(problem, a)
        x = np.array([1.1, .45, 1.7])
        _, g = f(x)
        eps = 1e-6
        fd = [(f(x+eps*e)[0]-f(x-eps*e)[0])/(2*eps) for e in np.eye(3)]
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-5)


def test_alpha_boundaries_and_constrained_never_beats_true_optimum(setup):
    fold, q0, h0, _ = setup
    row = cap.profile_participant(_job(fold, q0, h0, grid=(0., 1.)))
    assert [p['alpha_ck'] for p in row['points']] == [0., .2, 1.]
    assert all(p['status'] == 'ok' for p in row['points'])
    assert row['points'][0]['half_life']['infinite'] and row['points'][2]['half_life']['case'] == 'immediate_one_update'
    assert min(p['delta_nll'] for p in row['points']) == pytest.approx(0, abs=1e-12)
    # the saved theta here is not an optimum, so constrained fits beat it and are flagged as violations, never hidden
    assert row['violations'] and all(v['improvement_nll'] > 0 for v in row['violations'])


def test_reproducible_under_documented_seeds(setup):
    fold, q0, h0, _ = setup
    np.testing.assert_array_equal(cap.random_nuisance_starts(5), cap.random_nuisance_starts(5))
    assert not np.array_equal(cap.random_nuisance_starts(5), cap.random_nuisance_starts(6))
    a = cap.profile_participant(_job(fold, q0, h0))
    b = cap.profile_participant(_job(fold, q0, h0))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_session5_replay_is_causal(setup):
    fold, q0, h0, _ = setup
    theta = [.3, 1.4, .5, 1.1]
    _, base = replay_fold(fold, 'ck', theta, q0, h0)
    t = 20
    left = fold.validation_trials.left.copy(); left[t] = ~left[t]
    changed = SimpleNamespace(fit_trials=fold.fit_trials, validation_trials=AuthorsTrials(
        fold.validation_trials.pairs, left, fold.validation_trials.rewards))
    _, alt = replay_fold(changed, 'ck', theta, q0, h0)
    np.testing.assert_array_equal(base.probability_left[:t+1], alt.probability_left[:t+1])  # choice t only affects later rows
    assert not np.array_equal(base.probability_left[t+1:], alt.probability_left[t+1:])


def _fail(*args, **kwargs):
    return SimpleNamespace(success=False, fun=np.nan, x=np.full(3, np.nan), message='forced failure', nit=0, nfev=0)


def test_failed_optimization_cannot_enter_analysis(setup):
    fold, q0, h0, problem = setup
    points = cap.profile_curve(problem, (0., .5), .5, np.array([1., .5, 1.]), cap.random_nuisance_starts(1, 2), minimizer=_fail)
    assert all(p['status'] == 'failed' and 'deviance' not in p and len(p['failed_starts']) == p['starts'] for p in points)
    good = cap.profile_participant(_job(fold, q0, h0, grid=cap.GRID, n_random=1))
    bad = copy.deepcopy(good); bad['participant'] = 'broken'
    bad['points'][3] = dict(alpha_ck=bad['points'][3]['alpha_ck'], anchor=False, status='failed', starts=3, successful_starts=0, failed_starts=[])
    other = copy.deepcopy(good); other['participant'] = 'other'
    kept, excluded = cap.usable([good, bad, other])
    assert [r['participant'] for r in kept] == ['synthetic', 'other'] and excluded == ['broken']
    s = cap.summarize([good, bad, other], draws=50)
    assert s['analysed'] == 2 and s['excluded_incomplete_profiles'] == ['broken'] and s['optimization']['failed_points'] == 1


def test_training_fold_rejects_final_test_rows():
    frame = pd.DataFrame(dict(ID=['a']*2, phase=['training', 'test'], session=[1, 5], trial=[1, 1]))
    with pytest.raises(ValueError, match='final test is forbidden'):
        make_fold(frame, 'a', 5)


class _Guarded(dict):
    def __getitem__(self, key):
        if key == 'data_4_analysis_RP_test':
            raise AssertionError('final-test table accessed')
        return super().__getitem__(key)


def _script():
    spec = importlib.util.spec_from_file_location('run_ck_alpha_profile', ROOT/'scripts/run_ck_alpha_profile.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not RAW.exists(), reason='Raw .RData unavailable locally.')
def test_real_baseline_reproduces_saved_fit_without_touching_final_test(monkeypatch):
    mod = _script()
    real = mod.load_rdata(RAW)
    monkeypatch.setattr(mod, 'load_rdata', lambda path: _Guarded(real))
    folds = json.loads((ROOT/'docs/history_diagnostics_results.json').read_text())
    p = next(r['participant'] for r in folds['rows'] if r['session'] == 5)
    (job,) = mod.build_jobs(RAW, folds, [p], 1)
    participant, index, fold, q0, h0, saved, _, _ = job
    out, _ = cap.score(fold, saved['theta'], q0, h0)
    assert abs(out['train_deviance']-saved['optimization']['optimizer_objective_deviance']) < 1e-7
    assert abs(out['s5_nll']-saved['validation']['nll']) < 1e-9
    assert fold.fit_sessions.max() == 4 and len(fold.validation_trials.left) == saved['validation']['n']
