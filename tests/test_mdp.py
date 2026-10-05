from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp

from reward_pairs.mdp import PairMDP, SoftValueError, build_empirical_mdp, soft_value_iteration
from reward_pairs.simulation import left_probability
from reward_pairs.trajectories import Trial


def terminal_mdp(states):
    transitions = np.zeros((len(states), 2, len(states) + 1))
    transitions[:, :, -1] = 1
    return PairMDP(tuple(states), transitions)


def test_orientation_features_and_rewards():
    mdp = terminal_mdp([(4, 6), (6, 4)])
    assert mdp.states == ((4, 6), (6, 4))
    np.testing.assert_array_equal(mdp.features[0, 0], np.eye(8)[3])
    np.testing.assert_array_equal(mdp.features[0, 1], np.eye(8)[5])
    np.testing.assert_array_equal(mdp.features[1, 0], np.eye(8)[5])
    np.testing.assert_array_equal(mdp.features.sum(axis=2), 1)
    np.testing.assert_array_equal(mdp.rewards(np.arange(1, 9)), [[4, 6], [6, 4]])


def test_empirical_counts_boundaries_gaps_and_human_action_independence():
    trials = [Trial("p", 1, 1, 1, 2, "left"), Trial("p", 1, 2, 2, 1, "right"),
              Trial("p", 1, 4, 1, 2, None), Trial("p", 2, 1, 3, 4, "left"),
              Trial("q", 1, 1, 4, 3, "right")]
    mdp = build_empirical_mdp(list(reversed(trials)))
    index = {s: i for i, s in enumerate(mdp.states)}
    p = mdp.transitions[:, 0]
    assert p[index[(1, 2)], index[(2, 1)]] == 0.5
    assert p[index[(1, 2)], -1] == 0.5
    assert p[index[(2, 1)], -1] == 1  # gap: do not jump from trial 2 to 4
    assert p[index[(3, 4)], -1] == 1  # no cross-participant transition
    assert p[index[(1, 2)], index[(3, 4)]] == 0  # no cross-session transition
    np.testing.assert_allclose(mdp.transitions.sum(axis=2), 1, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(mdp.transitions[:, 0], mdp.transitions[:, 1])
    changed = build_empirical_mdp([replace(t, action="right") for t in trials])
    np.testing.assert_array_equal(changed.transitions, mdp.transitions)
    assert mdp.summary.observations == 5
    assert mdp.summary.consecutive_edges == 1
    assert mdp.summary.session_endings == mdp.summary.sessions == 3
    assert mdp.summary.gap_endings == 1


def test_invalid_scaffold_rejected():
    with pytest.raises(ValueError, match="empty"):
        build_empirical_mdp([])
    trial = Trial("p", 1, 1, 1, 2, None)
    with pytest.raises(ValueError, match="Duplicate"):
        build_empirical_mdp([trial, trial])


@pytest.mark.parametrize("bad", [np.nan, -0.1, 0.5])
def test_invalid_transition_probabilities_rejected(bad):
    with pytest.raises(ValueError, match="probabilities|sum to one"):
        PairMDP(((1, 2),), np.array([[[0, bad], [0, 1]]]))


def test_terminal_has_no_entropy_and_two_step_values_are_hand_checkable():
    transitions = np.zeros((2, 2, 3))
    transitions[0, :, 1] = 1
    transitions[1, :, 2] = 1
    mdp = PairMDP(((1, 2), (3, 4)), transitions)
    weights = np.array([0, 1, 2, 3, 0, 0, 0, 0])
    beta, gamma = 0.7, 0.9
    solution = soft_value_iteration(mdp, weights, beta=beta, gamma=gamma)
    v1 = logsumexp(beta * weights[2:4]) / beta
    q0 = weights[:2] + gamma * v1
    np.testing.assert_allclose(solution.q_values[1], weights[2:4], atol=1e-12)
    np.testing.assert_allclose(solution.q_values[0], q0, atol=1e-12)
    np.testing.assert_allclose(solution.values, [logsumexp(beta * q0) / beta, v1], atol=1e-12)


@pytest.mark.parametrize("gamma", [0, 0.5, 0.95, 0.99])
@pytest.mark.parametrize("kernel", ["terminal", "self", "cycle", "mixed"])
def test_continuation_cancellation_gamma_and_kernel_invariance(gamma, kernel):
    states = ((1, 8), (8, 1), (2, 3), (4, 6))
    n = len(states)
    p = np.zeros((n, n + 1))
    if kernel == "terminal":
        p[:, -1] = 1
    elif kernel == "self":
        p[:, :n] = np.eye(n)
    elif kernel == "cycle":
        p[np.arange(n), (np.arange(n) + 1) % n] = 1
    else:
        p[:] = np.random.default_rng(73).dirichlet(np.ones(n + 1), size=n)
    mdp = PairMDP(states, np.repeat(p[:, None, :], 2, axis=1))
    weights = np.array([0, 2, 2, -1, 0.7, 1.1, -2, 4])
    solution = soft_value_iteration(mdp, weights, beta=0.8, gamma=gamma)
    np.testing.assert_allclose(solution.policy.sum(axis=1), 1, atol=1e-14)
    continuation = solution.q_values - mdp.rewards(weights)
    np.testing.assert_allclose(continuation[:, 0], continuation[:, 1], atol=1e-12)
    np.testing.assert_allclose(np.diff(solution.q_values, axis=1), np.diff(mdp.rewards(weights), axis=1), atol=1e-12)
    np.testing.assert_allclose(solution.policy[:, 0], left_probability(states, weights, beta=0.8), atol=1e-12, rtol=0)
    assert solution.bellman_residual <= 1e-10


def test_large_rewards_keep_log_probabilities_finite():
    mdp = terminal_mdp([(1, 8), (8, 1)])
    solution = soft_value_iteration(mdp, [-1000, 0, 0, 0, 0, 0, 0, 1000], beta=100)
    assert np.isfinite(solution.log_policy).all()
    np.testing.assert_array_equal(solution.policy, [[0, 1], [1, 0]])
    np.testing.assert_allclose(solution.log_policy[0], [-200_000, 0])


def test_action_dependent_negative_control_proves_solver_uses_dynamics():
    # Equal immediate rewards at state 0, but only left reaches valuable state 1.
    p = np.zeros((2, 2, 3))
    p[0, 0, 1] = 1
    p[0, 1, 2] = 1
    p[1, :, 2] = 1
    mdp = PairMDP(((1, 2), (3, 4)), p)
    weights = [0, 0, 2, 2, 0, 0, 0, 0]
    assert soft_value_iteration(mdp, weights, gamma=0).policy[0, 0] == 0.5
    assert soft_value_iteration(mdp, weights, gamma=0.9).policy[0, 0] > 0.9


def test_nonconvergence_is_explicit():
    mdp = PairMDP(((1, 2),), np.array([[[1, 0], [1, 0]]]))
    with pytest.raises(SoftValueError, match="failed after"):
        soft_value_iteration(mdp, np.zeros(8), max_iterations=1)


@pytest.mark.parametrize("kwargs", [{"beta": 0}, {"gamma": 1}, {"gamma": -0.1}, {"gamma": np.nan}])
def test_invalid_planning_configuration(kwargs):
    with pytest.raises(ValueError):
        soft_value_iteration(terminal_mdp([(1, 2)]), np.zeros(8), **kwargs)
