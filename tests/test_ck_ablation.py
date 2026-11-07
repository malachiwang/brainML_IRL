"""Experiment 2: kernel interventions are exact, isolated, and correctly summarized."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from reward_pairs.authors_models import AuthorsTrials, evaluate as authors_evaluate
from reward_pairs.ck_ablation import (
    CONDITIONS, NEUTRAL_K, decompose, effect_summary, factorial, kernel_replay, pair_classes,
    paired_bootstrap, participant_summary, run_conditions, terciles,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/02_comp_mod_RP_task_data_in.RData'
THETA = [.3, 1.7, .4, 1.2]  # alpha_CK, beta_CK, alpha_RL, beta_RL


@pytest.fixture
def trials():
    rng = np.random.default_rng(3)
    trained = np.array([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
    pairs = trained[rng.integers(0, 8, 60)]
    pairs[::2] = pairs[::2, ::-1]
    values = np.array([1, 2, 2, 3, 3, 4, 4, 5.])
    return AuthorsTrials(pairs, rng.random(60) < .6, values[pairs])


@pytest.fixture
def states():
    rng = np.random.default_rng(4)
    return rng.uniform(0, 3, 8), rng.uniform(0, 1, 8)


def test_no_intervention_is_exactly_authors_evaluate(trials, states):
    q0, h0 = states
    for rewards in (trials.rewards, None):
        t = AuthorsTrials(trials.pairs, trials.left, rewards)
        a, b = authors_evaluate(t, 19, THETA, q0, h0), kernel_replay(t, THETA, q0, h0)
        for field in ('probability_left', 'log_loss', 'q', 'h'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_freeze_holds_kernel_and_leaves_q_untouched(trials, states):
    q0, h0 = states
    out = run_conditions(trials, THETA, q0, h0)  # asserts Q identical across all conditions
    np.testing.assert_array_equal(out['C1'].h, np.tile(h0, (len(trials.left)+1, 1)))
    assert not np.allclose(out['C0'].h[-1], h0)
    np.testing.assert_array_equal(out['C1'].q, authors_evaluate(trials, 19, THETA, q0, h0).q)


def test_reset_starts_at_neutral_and_then_updates(trials, states):
    q0, h0 = states
    c2 = kernel_replay(trials, THETA, q0, h0, reset_to=NEUTRAL_K)
    np.testing.assert_array_equal(c2.h[0], np.full(8, NEUTRAL_K))
    reference = authors_evaluate(trials, 19, THETA, q0, np.full(8, NEUTRAL_K))
    np.testing.assert_array_equal(c2.probability_left, reference.probability_left)
    # Any common reset value gives the same first prediction (only K differences matter).
    first = [kernel_replay(trials, THETA, q0, h0, reset_to=c).probability_left[0] for c in (0., .5, 1., 3.)]
    assert np.ptp(first) < 1e-15


def test_c3_removes_only_kernel_term_and_equals_reset_plus_freeze(trials, states):
    q0, h0 = states
    c3 = kernel_replay(trials, THETA, q0, h0, kernel_weight=0.)
    both = kernel_replay(trials, THETA, q0, h0, reset_to=NEUTRAL_K, freeze=True)
    np.testing.assert_allclose(c3.probability_left, both.probability_left, rtol=0, atol=1e-15)
    t = np.arange(len(trials.left)); l, r = trials.pairs.T
    np.testing.assert_allclose(c3.probability_left, expit(THETA[3]*(c3.q[t, l]-c3.q[t, r])), rtol=0, atol=1e-15)
    reward, kernel = decompose(trials, c3, THETA, 'C3')
    assert not kernel.any() and reward.any()


def test_decomposition_matches_probabilities(trials, states):
    q0, h0 = states
    out = run_conditions(trials, THETA, q0, h0)
    for name in CONDITIONS:
        reward, kernel = decompose(trials, out[name], THETA, name)
        np.testing.assert_allclose(expit(reward+kernel), out[name].probability_left, atol=1e-12)


def test_pair_classes_and_terciles():
    test_pairs = np.array([[0, 2], [2, 1], [3, 4], [0, 7], [5, 6], [6, 4]])  # 1-3, 3-2, 4-5, 1-8, 6-7, 7-5
    np.testing.assert_array_equal(pair_classes(test_pairs, 'test'),
                                  ['trained_pair', 'novel_equal', 'novel_equal', 'novel_unequal', 'novel_equal', 'trained_pair'])
    np.testing.assert_array_equal(pair_classes(np.array([[0, 2], [0, 1]]), 'validation'), ['high_freq_pair', 'low_freq_pair'])
    with pytest.raises(ValueError, match='untrained'):
        pair_classes(np.array([[1, 2]]), 'validation')
    lab = terciles(10)
    assert list(np.bincount(lab)) == [4, 3, 3] and np.all(np.diff(lab) >= 0)


def _row(losses):
    return dict(n=10, conditions={c: dict(loss=v, nll=10*v) for c, v in zip(CONDITIONS, losses)})


def test_factorial_algebra_and_signs():
    rows = [_row([.30, .31, .32, .40]), _row([.35, .34, .36, .45]), _row([.40, .43, .41, .42])]
    fac = factorial(rows, draws=200)
    L = np.array([[r['conditions'][c]['loss'] for c in CONDITIONS] for r in rows])
    assert fac['total_kernel_value']['mean'] == pytest.approx((L[:, 3]-L[:, 0]).mean())
    assert fac['updating_given_carried_state']['mean'] == pytest.approx((L[:, 1]-L[:, 0]).mean())
    assert fac['interaction']['mean'] == pytest.approx(((L[:, 3]-L[:, 1])-(L[:, 2]-L[:, 0])).mean())
    total = fac['updating_given_carried_state']['mean']+fac['carried_state_without_updating']['mean']
    assert total == pytest.approx(fac['total_kernel_value']['mean'])
    assert fac['updating_given_carried_state']['hurts'] == 2 and fac['updating_given_carried_state']['helps'] == 1
    assert fac['updating_given_carried_state']['share_of_total'] == pytest.approx((L[:, 1]-L[:, 0]).mean()/(L[:, 3]-L[:, 0]).mean())


def test_bootstrap_is_paired_and_deterministic():
    a = np.arange(20.); b = a+1
    one = paired_bootstrap([a, b], lambda x, y: (y-x).mean(axis=1), draws=300)
    assert one['ci95'] == pytest.approx([1., 1.])  # paired resampling keeps the constant difference
    assert one == paired_bootstrap([a, b], lambda x, y: (y-x).mean(axis=1), draws=300)
    s = effect_summary([0., 1e-9, .2, -.1], draws=100)
    assert (s['hurts'], s['helps'], s['tied']) == (1, 1, 2)


def test_participant_summary_contributions(trials, states):
    q0, h0 = states
    out = run_conditions(trials, THETA, q0, h0)
    s = participant_summary(trials, THETA, out, 'validation')
    assert s['conditions']['C3']['contributions']['all']['median_abs_kernel'] == 0
    assert s['conditions']['C0']['loss'] == pytest.approx(out['C0'].log_loss.mean())
    assert sum(s['class_counts'].values()) == len(trials.left)


@pytest.mark.skipif(not RAW.exists(), reason='Raw .RData unavailable locally.')
def test_real_participant_reproduces_saved_session5_and_test():
    spec = importlib.util.spec_from_file_location('run_ck_ablation', ROOT/'scripts/run_ck_ablation.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    folds = json.loads((ROOT/'docs/history_diagnostics_results.json').read_text())
    authors = json.loads((ROOT/'docs/authors_baseline_results.json').read_text())
    row = next(r for r in folds['rows'] if r['session'] == 5)
    p = row['participant']
    v = mod.validation_job((p, RAW, row['models']['ck'], row['models']['rl']))
    assert v['checks']['c0_validation_nll'] < 1e-9 and v['checks']['carried_k'] < 1e-12 and v['checks']['fit_deviance'] < 1e-7
    rec = {(r['participant'], r['variant'], r['model']): r for r in authors['rows']}
    t = mod.test_job((p, RAW, rec[p, 'released_code_intended', 19], rec[p, 'released_code_intended', 3]))
    assert t['checks']['c0_test_deviance'] < 1e-7
