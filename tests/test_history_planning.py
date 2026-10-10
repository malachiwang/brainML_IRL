from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logsumexp

from reward_pairs.history import ChoiceHistory
from reward_pairs.history_mdp import HistoryMDP, HistoryState, reachable_state_audit
from reward_pairs.history_planning import PlanningBatch, planning_rollouts, soft_plan
from reward_pairs.history_irl import PlanningLikelihood, demonstration_states, predict_planning, recover_planning
from reward_pairs.history_simulation import independent_rollouts, history_probability
from reward_pairs.recovery import RecoveryError
from reward_pairs.simulation import DEFAULT_UTILITIES
from reward_pairs.trajectories import Trial

TOY = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (1, 8)]


def test_action_driven_counts_presentations_pair_progression_and_terminal():
    m = HistoryMDP([(2, 3), (4, 2)], chronology=[(1, 159), (2, 3)])
    left, right = [m.transition(m.initial, a) for a in ('left', 'right')]
    assert left != right and left.chosen[1] == 1 and right.chosen[2] == 1
    assert sum(left.chosen) == sum(right.chosen) == 1
    np.testing.assert_array_equal(m.presented[1], [0, 1, 1, 0, 0, 0, 0, 0])
    assert tuple(m.pairs[left.t]) == (4, 2)
    assert m.history(left)[1] == .5  # carries across session and retained gap
    end = m.transition(left, 'right')
    assert end.chosen[1] == 2
    assert soft_plan(m, end, np.zeros(8), alpha=1).value == 0
    with pytest.raises(ValueError, match='nonterminal'):
        m.transition(end, 'left')


def test_human_actions_not_read():
    trials = [Trial('p', 1, i + 1, *pair, 'left') for i, pair in enumerate(TOY)]
    a = HistoryMDP.from_trials(trials)
    b = HistoryMDP.from_trials([replace(t, action='right') for t in trials])
    np.testing.assert_array_equal(a.pairs, b.pairs)
    np.testing.assert_array_equal(a.presented, b.presented)


@pytest.mark.parametrize('gamma,h', [(0., 3), (.95, 0), (1., 0)])
def test_myopic_limit_and_reference_shift(gamma, h):
    m = HistoryMDP([(2, 3), (2, 1), (3, 2), (2, 8)])
    s = m.transition(m.initial, 'left')
    history = ChoiceHistory(m.presented[s.t], np.array(s.chosen))
    expected = history_probability(2, 1, history, DEFAULT_UTILITIES, alpha=.7, beta=.8)
    p = soft_plan(m, s, DEFAULT_UTILITIES, alpha=.7, beta=.8, gamma=gamma, lookahead=h)
    assert np.exp(p.log_policy[0]) == pytest.approx(expected, abs=1e-14)
    shifted = soft_plan(m, s, np.array(DEFAULT_UTILITIES) + 19, alpha=.7, beta=.8, gamma=gamma, lookahead=h)
    np.testing.assert_allclose(shifted.log_policy, p.log_policy, atol=1e-13)


def test_hand_calculated_two_step_non_cancellation_and_future_schedule():
    # Current equal rewards; left increases stimulus 2's next reward by 1
    # relative to right. Stimulus 1 is unseen at t=1 (H=0).
    m = HistoryMDP([(2, 3), (2, 1)])
    p = soft_plan(m, m.initial, np.zeros(8), alpha=2, gamma=.95, lookahead=1)
    future_values = np.array([np.logaddexp(1., 0.), np.logaddexp(-1., 0.)])
    np.testing.assert_allclose(p.q, .95 * future_values, atol=1e-14)
    assert p.q[0] - p.q[1] == pytest.approx(.95)
    assert np.exp(p.log_policy[0]) == pytest.approx(expit(.95))
    assert soft_plan(m, m.initial, np.zeros(8), alpha=2, gamma=0, lookahead=1).log_policy[0] == pytest.approx(-np.log(2))
    reversed_future = HistoryMDP([(2, 3), (3, 1)])
    reverse = soft_plan(reversed_future, reversed_future.initial, np.zeros(8), alpha=2, lookahead=1)
    assert np.exp(reverse.log_policy[0]) < .5


def test_three_step_planning_distinguishable_from_myopic_and_horizon_respected():
    m = HistoryMDP([(2, 3), (2, 1), (2, 1)])
    p = soft_plan(m, m.initial, np.zeros(8), alpha=2, lookahead=2)
    assert np.exp(p.log_policy[0]) > .75
    a = soft_plan(m, m.initial, np.zeros(8), alpha=2, lookahead=1)
    other = HistoryMDP([(2, 3), (2, 1), (3, 8)])
    b = soft_plan(other, other.initial, np.zeros(8), alpha=2, lookahead=1)
    np.testing.assert_array_equal(a.log_policy, b.log_policy)
    assert not np.allclose(p.log_policy, soft_plan(other, other.initial, np.zeros(8), alpha=2, lookahead=2).log_policy)


@pytest.mark.parametrize('h', [0, 1, 3, 5])
@pytest.mark.parametrize('alpha', [0., -.8, 1.])
def test_scalar_memoized_unmemoized_and_batch_policies_gradients(h, alpha):
    m = HistoryMDP([(2, 3)] * 6 + [(2, 1)])
    s = m.transition(m.initial, 'left')
    w = np.array(DEFAULT_UTILITIES) - 1
    a = soft_plan(m, s, w, alpha=alpha, lookahead=h, memoize=True)
    b = soft_plan(m, s, w, alpha=alpha, lookahead=h, memoize=False)
    np.testing.assert_array_equal(a.q, b.q)
    if h >= 3:
        assert a.cache_hits > 0 and a.evaluated_states < b.evaluated_states
    batch = PlanningBatch(m, [s.t], [s.chosen], lookahead=h)
    theta = np.r_[w[1:], alpha]
    q, lp, jac = batch.evaluate(theta, gradient=True)
    np.testing.assert_allclose(q[0], a.q, atol=1e-13)
    np.testing.assert_allclose(lp[0], a.log_policy, atol=1e-13)
    assert np.exp(lp).sum() == pytest.approx(1)
    if alpha == 0:
        assert np.exp(lp[0, 0]) == pytest.approx(.5)
    for i, delta in enumerate(1e-5 * np.eye(8)):
        diff = (batch.evaluate(theta + delta)[1] - batch.evaluate(theta - delta)[1]) / 2e-5
        np.testing.assert_allclose(jac[..., i], diff, atol=1e-9)


def test_terminal_padding_and_large_scores_beta():
    m = HistoryMDP([(1, 8), (8, 1)])
    end = m.transition(m.initial, 'left')
    b = PlanningBatch(m, [0, 1], [m.initial.chosen, end.chosen], lookahead=5)
    w = np.array([0, 0, 0, 0, 0, 0, 0, 1000.])
    q, lp, _ = b.evaluate(np.r_[w[1:], 0], beta=100)
    assert np.isfinite(lp).all()
    np.testing.assert_array_equal(np.exp(lp), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(q[1], [1000, 0])
    for beta in (.1, 1., 2.):
        p = soft_plan(m, m.initial, w / 1000, alpha=0, beta=beta, lookahead=1)
        assert np.exp(p.log_policy[0]) == pytest.approx(expit(-beta))


def test_batch_memory_guard_and_invalid_inputs():
    m = HistoryMDP(TOY)
    with pytest.raises(ValueError, match='allocate'):
        PlanningBatch(m, [0], [m.initial.chosen], lookahead=7, max_nodes=100)
    with pytest.raises(ValueError, match='0..10'):
        soft_plan(m, m.initial, np.zeros(8), alpha=1, lookahead=783)
    with pytest.raises(ValueError, match='inconsistent'):
        m.history(HistoryState(0, (1,) * 8))


def test_simulation_reproducibility_h_zero_old_simulator_and_q_probabilities():
    trials = [Trial('p', 1, i + 1, *pair, None) for i, pair in enumerate(TOY)]
    m = HistoryMDP.from_trials(trials)
    a = planning_rollouts(m, 4, alpha=1, seed=71, lookahead=0)
    b = independent_rollouts(trials, 4, alpha=1, seed=71)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.choices, y.choices)
        np.testing.assert_allclose(x.probabilities, y.probabilities, atol=1e-15)
    planned = planning_rollouts(m, 4, alpha=1, seed=71, lookahead=3)
    again = planning_rollouts(m, 4, alpha=1, seed=71, lookahead=3)
    pd.testing.assert_frame_equal(planned[0].choices, again[0].choices)
    _, times, counts = demonstration_states(m, planned[0].choices)
    expected = [np.exp(soft_plan(m, HistoryState(t, tuple(c)), DEFAULT_UTILITIES, alpha=1, lookahead=3).log_policy[0]) for t, c in zip(times, counts)]
    np.testing.assert_allclose(expected, planned[0].probabilities, atol=1e-14)


def test_future_and_current_observed_actions_do_not_affect_current_policy():
    m = HistoryMDP(TOY)
    data = planning_rollouts(m, 1, alpha=1, seed=2)[0].choices
    changed = data.copy()
    changed.loc[3:, 'synthetic_action'] = np.where(data.loc[3:, 'synthetic_action']=='left', 'right', 'left')
    theta = np.r_[np.array(DEFAULT_UTILITIES)[1:] - 1, 1]
    a = PlanningLikelihood(m, data).probabilities(theta)[0]
    b = PlanningLikelihood(m, changed).probabilities(theta)[0]
    np.testing.assert_array_equal(a[:4], b[:4])
    # Supplied truth, history and human fields cannot enter the likelihood.
    poisoned = data.assign(action='right', reward1=999, alpha_true=-100, left_history=1e9)
    np.testing.assert_array_equal(a, PlanningLikelihood(m, poisoned).probabilities(theta)[0])


def test_objective_gradient_and_missing_prefix_rejected():
    m = HistoryMDP(TOY)
    data = pd.concat([r.choices for r in planning_rollouts(m, 10, alpha=.5, seed=9)])
    likelihood = PlanningLikelihood(m, data, lookahead=7)
    theta = np.r_[np.arange(7) * .3, -.7]
    loss, grad = likelihood.objective(theta)
    assert np.isfinite(loss)
    for i, d in enumerate(1e-5*np.eye(8)):
        numerical = (likelihood.objective(theta+d)[0] - likelihood.objective(theta-d)[0])/2e-5
        assert grad[i] == pytest.approx(numerical, abs=1e-9)
    with pytest.raises(ValueError, match='prefix'):
        PlanningLikelihood(m, data.iloc[1:])


@pytest.mark.parametrize('alpha', [-1., 0., 1.])
def test_exact_eight_trial_matched_recovery(alpha):
    m = HistoryMDP(TOY)
    data = pd.concat([r.choices for r in planning_rollouts(m, 1200, alpha=alpha, seed=31, lookahead=7)], ignore_index=True)
    fit = recover_planning(m, data, lookahead=7)
    assert fit.utilities.loc[1] == 0
    assert abs(fit.alpha - alpha) < .3
    np.testing.assert_allclose(fit.utilities, np.array(DEFAULT_UTILITIES)-1, atol=.35)
    assert fit.information_min_eigenvalue > 0
    assert predict_planning(m, fit, data).log_loss.sum() == pytest.approx(fit.negative_log_likelihood, abs=1e-8)


def test_optimizer_failure_and_unidentified_problem_explicit(monkeypatch):
    m = HistoryMDP([(1, 2)] * 4)
    data = pd.concat([r.choices for r in planning_rollouts(m, 30, alpha=1)])
    with pytest.raises(RecoveryError, match='Unidentified'):
        recover_planning(m, data)
    monkeypatch.setattr('reward_pairs.history_irl.minimize', lambda *a, **k: SimpleNamespace(
        success=False, fun=1., x=np.zeros(8), message='deliberate failure'))
    with pytest.raises(RecoveryError, match='failed'):
        recover_planning(m, data)


def test_reachable_states_and_tree_bound_are_not_path_counts():
    m = HistoryMDP([(1, 2)] * 6)
    audit = reachable_state_audit(m)
    assert audit['exact_prefix_layer_counts'] == list(range(1, 8))
    m = HistoryMDP(TOY)
    audit = reachable_state_audit(m)
    assert audit['preterminal_lower_bound'] == 2**7
    assert audit['exact_prefix_layer_counts'][-2] >= audit['preterminal_lower_bound']
    assert reachable_state_audit(m, cap=10)['stopped_next_layer_at'] == 11


def test_session_five_labels_cannot_change_training_fit_or_earlier_predictions():
    pairs = TOY * 20
    m = HistoryMDP(pairs, chronology=[(i//32+1, i%32+1) for i in range(len(pairs))])
    data = pd.concat([r.choices for r in planning_rollouts(m, 10, alpha=1, seed=14)], ignore_index=True)
    altered = data.copy()
    altered.loc[altered.session == 5, 'synthetic_action'] = 'right'
    a = recover_planning(m, data, fit_sessions=[1,2,3,4])
    b = recover_planning(m, altered, fit_sessions=[1,2,3,4])
    np.testing.assert_array_equal(a.utilities, b.utilities)
    assert a.alpha == b.alpha
    pa, pb = predict_planning(m, a, data), predict_planning(m, a, altered)
    mask = (pa.session<5) | ((pa.session==5) & (pa.trial==1))
    np.testing.assert_array_equal(pa.loc[mask,'probability_left'], pb.loc[mask,'probability_left'])


def test_arbitrary_hidden_rewards_and_h_zero_estimator_equivalence():
    from reward_pairs.history_recovery import recover_history
    m = HistoryMDP(TOY)
    truth = np.array([0., .3, -.2, .7, 1.1, -.4, .5, 1.4])
    data = pd.concat([r.choices for r in planning_rollouts(m, 1500, utilities=truth, alpha=-.6,
                        beta=.8, gamma=.7, lookahead=3, seed=23)], ignore_index=True)
    fit = recover_planning(m, data, beta=.8, gamma=.7, lookahead=3)
    np.testing.assert_allclose(fit.utilities, truth, atol=.2)
    assert abs(fit.alpha + .6)<.3
    static_h = recover_history(data, beta=.8)
    h0 = recover_planning(m, data, beta=.8, lookahead=0)
    np.testing.assert_allclose(h0.utilities, static_h.utilities, atol=1e-5, rtol=0)
    assert h0.alpha == pytest.approx(static_h.alpha, abs=1e-5)
    assert h0.negative_log_likelihood == pytest.approx(static_h.negative_log_likelihood, abs=1e-8)


def test_nonfinite_optimizer_and_iteration_cap_fail_explicitly(monkeypatch):
    m = HistoryMDP(TOY)
    data = pd.concat([r.choices for r in planning_rollouts(m, 50, alpha=1, seed=7)])
    with pytest.raises(RecoveryError, match='failed'):
        recover_planning(m, data, max_iterations=1)
    monkeypatch.setattr('reward_pairs.history_irl.minimize', lambda *a, **k: SimpleNamespace(
        success=True, fun=np.nan, x=np.zeros(8), message='nonfinite result'))
    with pytest.raises(RecoveryError, match='failed'):
        recover_planning(m, data)
