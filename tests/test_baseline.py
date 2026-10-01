from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from reward_pairs.evaluation import compare_utilities, recovery_stability
from reward_pairs.recovery import RecoveryError, recover_utilities
from reward_pairs.simulation import DEFAULT_UTILITIES, left_probability, simulate_choices, simulate_trajectory
from reward_pairs.trajectories import Trial


@pytest.fixture(scope="module")
def large_choices():
    # All 28 comparisons, both orientations, 56,000 fresh choices.
    pairs = np.array(list(combinations(range(1, 9), 2)))
    states = np.tile(np.vstack([pairs, pairs[:, ::-1]]), (1000, 1))
    return simulate_choices(states, seed=41, beta=0.7)


def test_probabilities_equal_values_orientation_and_temperature():
    states = [(2, 3), (1, 8), (8, 1)]
    probabilities = left_probability(states, beta=0.5)
    np.testing.assert_allclose(probabilities, [0.5, 1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))])
    np.testing.assert_allclose(left_probability(states, beta=0), 0.5)
    np.testing.assert_allclose(left_probability([(1, 8), (8, 1)], beta=1e6), [0, 1])


def test_fixed_seed_reproducibility_and_ignoring_human_actions():
    states = [(2, 3), (1, 8), (6, 4)] * 300
    trajectory = [Trial("p", 1, i + 1, left, right, "left") for i, (left, right) in enumerate(states)]
    altered = [replace(t, action=None) for t in trajectory]
    first = simulate_trajectory(trajectory, seed=123)
    pd.testing.assert_frame_equal(first, simulate_trajectory(altered, seed=123))
    assert not first.equals(simulate_trajectory(trajectory, seed=124))
    assert list(first.columns) == ["left_stimulus", "right_stimulus", "synthetic_action"]


def test_empirical_choices_follow_probability():
    choices = simulate_choices([(4, 2)] * 30_000, beta=0.7, seed=17)
    observed = choices.synthetic_action.eq("left").mean()
    assert abs(observed - 1 / (1 + np.exp(-0.7))) < 0.015


def test_recovery_on_large_stochastic_sample(large_choices):
    fit = recover_utilities(large_choices, beta=0.7)
    truth = np.array(DEFAULT_UTILITIES) - DEFAULT_UTILITIES[0]
    np.testing.assert_allclose(fit.utilities, truth, atol=0.12)
    assert fit.utilities.loc[1] == 0
    assert fit.n_trials == 56_000
    assert fit.negative_log_likelihood > 0
    _, metrics = compare_utilities(fit)
    assert metrics["strict_order_agreement"] == 1
    assert metrics["pearson"] > 0.995
    assert metrics["mae"] < 0.07
    assert metrics["true_tie_mean_gap"] < 0.1


def test_reference_and_fixed_beta_identification(large_choices):
    first = recover_utilities(large_choices, beta=0.7)
    eighth = recover_utilities(large_choices, beta=0.7, reference=8)
    assert eighth.utilities.loc[8] == 0
    np.testing.assert_allclose(eighth.utilities, first.utilities - first.utilities.loc[8], atol=1e-5)
    # A different fixed beta rescales utilities but leaves choice probabilities.
    scaled = recover_utilities(large_choices, beta=1.0)
    np.testing.assert_allclose(scaled.utilities, 0.7 * first.utilities, atol=1e-5)
    pairs = [(1, 8), (3, 5), (4, 6)]
    shifted_truth = np.array(DEFAULT_UTILITIES) + 100
    np.testing.assert_allclose(left_probability(pairs, shifted_truth), left_probability(pairs))


def test_disconnected_design_cannot_identify_all_stimuli():
    choices = simulate_choices([(1, 2)] * 1000, seed=7)
    with pytest.raises(RecoveryError, match="disconnected"):
        recover_utilities(choices)


def test_separation_has_no_finite_mle():
    choices = pd.DataFrame({"left_stimulus": range(1, 8), "right_stimulus": range(2, 9), "synthetic_action": "right"})
    with pytest.raises(RecoveryError, match="no finite"):
        recover_utilities(choices)


def test_recovery_rejects_missing_actions_and_zero_beta(large_choices):
    choices = large_choices.copy()
    choices.loc[0, "synthetic_action"] = None
    with pytest.raises(ValueError, match="missing choices"):
        recover_utilities(choices)
    with pytest.raises(ValueError, match="beta"):
        recover_utilities(large_choices, beta=0)


def test_multiple_seeds_and_more_trials():
    states = list(combinations(range(1, 9), 2)) * 10
    runs = recovery_stability(states, seeds=list(range(8)), repeats=[1, 20], beta=0.7)
    assert len(runs) == 16 and runs.status.eq("ok").all()
    means = runs.groupby("repeats").mae.mean()
    assert means.loc[20] < means.loc[1]  # expectation across seeds, not monotonic per draw
    pd.testing.assert_frame_equal(
        runs, recovery_stability(states, seeds=list(range(8)), repeats=[1, 20], beta=0.7)
    )


def test_stability_reports_failed_seeds():
    runs = recovery_stability([(1, 2)], seeds=[0, 1], repeats=[1])
    assert len(runs) == 2
    assert runs.status.str.contains("disconnected").all()
