from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reward_pairs.evaluation import compare_utilities
from reward_pairs.irl import recover_rewards, soft_action_objective
from reward_pairs.mdp import PairMDP, build_empirical_mdp
from reward_pairs.recovery import RecoveryError, recover_utilities
from reward_pairs.simulation import left_probability, simulate_choices
from reward_pairs.trajectories import Trial


@pytest.fixture(scope="module")
def experiment():
    pairs = list(combinations(range(1, 9), 2))
    states = pairs + [(r, l) for l, r in pairs]
    trials = [Trial("fixture", 1, i + 1, l, r, None) for i, (l, r) in enumerate(states)]
    mdp = build_empirical_mdp(trials)
    choices = simulate_choices(np.tile(states, (500, 1)), seed=41, beta=0.7)
    return mdp, choices


def test_bellman_likelihood_gradient_against_finite_differences():
    # Deliberately action-dependent negative control tests the full derivative.
    p = np.array([[[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
                  [[0.4, 0.1, 0.5], [0.2, 0.6, 0.2]]])
    mdp = PairMDP(((1, 2), (3, 4)), p)
    weights = np.array([0, 0.3, -0.4, 0.7, 0, 0, 0, 0])
    counts = np.array([[12, 7], [6, 21]])
    _, gradient = soft_action_objective(weights, mdp, counts, beta=0.7, gamma=0.6)
    epsilon = 1e-5
    numerical = []
    for delta in epsilon * np.eye(8):
        plus = soft_action_objective(weights + delta, mdp, counts, beta=0.7, gamma=0.6)[0]
        minus = soft_action_objective(weights - delta, mdp, counts, beta=0.7, gamma=0.6)[0]
        numerical.append((plus - minus) / (2 * epsilon))
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("gamma", [0, 0.5, 0.95, 0.99])
def test_bt_irl_equivalence_and_reference(experiment, gamma):
    mdp, choices = experiment
    bt = recover_utilities(choices, beta=0.7)
    irl = recover_rewards(choices, mdp, beta=0.7, gamma=gamma)
    assert irl.utilities.loc[1] == 0
    np.testing.assert_allclose(irl.utilities, bt.utilities, atol=1e-5, rtol=0)
    np.testing.assert_allclose(irl.solution.policy[:, 0], left_probability(mdp.states, bt.utilities, beta=0.7), atol=1e-7, rtol=0)
    assert abs(irl.negative_log_likelihood - bt.negative_log_likelihood) < 1e-6
    _, metrics = compare_utilities(irl)
    assert metrics["strict_order_agreement"] == 1
    assert metrics["mae"] < 0.1


def test_arbitrary_hidden_truth_and_irrelevant_columns_cannot_leak(experiment):
    mdp, _ = experiment
    # Different ordering, no ties, including negative rewards; passed ONLY to simulator.
    truth = np.array([0, -0.3, 1.2, 0.5, -1, 1.7, 0.9, 2.3])
    choices = simulate_choices(np.tile(mdp.states, (500, 1)), seed=27, utilities=truth, beta=0.7)
    fit = recover_rewards(choices, mdp, beta=0.7)
    polluted = choices.assign(reward1=10_000, reward2=-10_000, reward_chosen=np.nan,
                              action="left", true_utilities="deliberately wrong")
    other = recover_rewards(polluted, mdp, beta=0.7)
    np.testing.assert_array_equal(fit.utilities, other.utilities)
    np.testing.assert_allclose(fit.utilities, truth, atol=0.15)
    assert abs(fit.utilities.loc[2] - fit.utilities.loc[3]) > 1


def test_identifiability_and_separation_fail_explicitly(experiment):
    mdp, _ = experiment
    with pytest.raises(RecoveryError, match="disconnected"):
        recover_rewards(simulate_choices([(1, 2)] * 100, seed=9), mdp)
    separated = pd.DataFrame({"left_stimulus": range(1, 8), "right_stimulus": range(2, 9), "synthetic_action": "right"})
    with pytest.raises(RecoveryError, match="no finite"):
        recover_rewards(separated, mdp)


def test_optimizer_failure_is_not_accepted(experiment):
    mdp, choices = experiment
    with pytest.raises(RecoveryError, match="optimization failed"):
        recover_rewards(choices, mdp, max_iterations=1)


def test_nonfinite_optimizer_output_is_rejected(experiment, monkeypatch):
    mdp, choices = experiment
    monkeypatch.setattr("reward_pairs.irl.minimize", lambda *a, **k: SimpleNamespace(
        success=True, x=np.full(7, np.nan), fun=1.0, message="invalid stub"))
    with pytest.raises(RecoveryError, match="optimization failed"):
        recover_rewards(choices, mdp)


def test_unknown_demonstration_states_rejected(experiment):
    mdp, choices = experiment
    p = np.zeros((1, 2, 2))
    p[:, :, -1] = 1
    with pytest.raises(ValueError, match="absent from the MDP"):
        recover_rewards(choices, PairMDP((mdp.states[0],), p))


def test_inference_rejects_action_dependent_kernel(experiment):
    mdp, choices = experiment
    p = mdp.transitions.copy()
    p[0, 0] = 0
    p[0, 0, -1] = 1
    with pytest.raises(ValueError, match="action-independent"):
        recover_rewards(choices, PairMDP(mdp.states, p))
