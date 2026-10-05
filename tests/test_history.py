from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from reward_pairs.history import ChoiceHistory, causal_history
from reward_pairs.history_simulation import history_probability, independent_rollouts, simulate_history
from reward_pairs.simulation import DEFAULT_UTILITIES, left_probability, simulate_trajectory
from reward_pairs.trajectories import Trial


def scaffold():
    return [Trial("p", 1, 1, 2, 3, "left"), Trial("p", 1, 3, 3, 2, "right"),
            Trial("p", 2, 1, 2, 3, None), Trial("p", 2, 2, 4, 6, "left")]


def choices():
    return pd.DataFrame({"rollout": [0] * 4, "session": [1, 1, 2, 2], "trial": [1, 3, 1, 2],
                         "left_stimulus": [2, 3, 2, 4], "right_stimulus": [3, 2, 3, 6],
                         "synthetic_action": ["left", "right", "right", "left"]})


def test_counters_initialization_and_orientation():
    h = ChoiceHistory()
    np.testing.assert_array_equal(h.centered, np.zeros(8))
    h.update(2, 3, "left")
    assert h.presented.tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert h.chosen.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    other = ChoiceHistory()
    other.update(3, 2, "right")
    np.testing.assert_array_equal(other.centered, h.centered)
    h.update(2, 2, "right")  # A stimulus is presented on one trial, not two.
    assert h.presented[1] == h.chosen[1] == 2


def test_history_precedes_current_action_and_carries_across_sessions_and_gaps():
    frame = causal_history(choices())
    np.testing.assert_array_equal(frame[["left_history", "right_history"]],
                                  [[0, 0], [-0.5, 0.5], [0.5, -0.5], [0, 0]])
    reset = causal_history(choices(), reset_history_each_session=True)
    np.testing.assert_array_equal(reset.loc[2, ["left_history", "right_history"]].to_numpy(dtype=float), [0, 0])


def test_future_and_current_choices_cannot_change_earlier_history():
    original = choices()
    altered = original.copy()
    altered.loc[1:, "synthetic_action"] = ["left", "left", "right"]
    a, b = causal_history(original), causal_history(altered)
    pd.testing.assert_frame_equal(a.loc[:1, ["left_history", "right_history"]], b.loc[:1, ["left_history", "right_history"]])
    assert a.left_history.iloc[2] != b.left_history.iloc[2]
    # Hand audit: trial 2 features reflect only trial 1, not its own right choice.
    assert a.left_history.iloc[1] == -0.5 and a.right_history.iloc[1] == 0.5


def test_supplied_history_and_truth_fields_ignored():
    frame = choices()
    polluted = frame.assign(left_history=123, right_history=-123, reward1=999, alpha_true=100)
    pd.testing.assert_frame_equal(causal_history(frame), causal_history(polluted))


@pytest.mark.parametrize("reset", [False, True])
def test_simulator_reproducible_and_alpha_zero_matches_existing_draws(reset):
    trajectory = scaffold()
    h = simulate_history(trajectory, alpha=0, seed=71, reset_history_each_session=reset)
    static = simulate_trajectory(trajectory, seed=71)
    np.testing.assert_array_equal(h.probabilities, left_probability([t.state for t in trajectory]))
    assert h.choices.synthetic_action.tolist() == static.synthetic_action.tolist()
    again = simulate_history([replace(t, action=None) for t in trajectory], alpha=0, seed=71,
                             reset_history_each_session=reset)
    pd.testing.assert_frame_equal(h.choices, again.choices)
    assert set(h.choices.columns) == {"rollout", "session", "trial", "left_stimulus", "right_stimulus", "synthetic_action"}


def test_history_direction_alpha_sign_and_beta():
    h = ChoiceHistory()
    # Equal utilities, chosen 2/6 versus 5/6.
    for i in range(6):
        h.update(2, 1, "left" if i < 2 else "right")
        h.update(3, 1, "left" if i < 5 else "right")
    assert h.rates[1] == 2 / 6 and h.rates[2] == 5 / 6
    assert history_probability(2, 3, h, DEFAULT_UTILITIES, alpha=1) < 0.5
    assert history_probability(2, 3, h, DEFAULT_UTILITIES, alpha=-1) > 0.5
    assert history_probability(2, 3, h, DEFAULT_UTILITIES, alpha=1, beta=0) == 0.5
    assert history_probability(2, 3, h, DEFAULT_UTILITIES, alpha=1, beta=2) < history_probability(2, 3, h, DEFAULT_UTILITIES, alpha=1)


def test_fresh_independent_histories_and_nested_rng_streams():
    one = independent_rollouts(scaffold(), 1, seed=4, alpha=1)
    many = independent_rollouts(scaffold(), 5, seed=4, alpha=1)
    pd.testing.assert_frame_equal(one[0].choices, many[0].choices)
    assert all(r.probabilities[0] == 0.5 for r in many)
    frame = causal_history(pd.concat([r.choices for r in many], ignore_index=True))
    np.testing.assert_array_equal(frame.groupby("rollout").first().left_history, np.zeros(5))
    assert many[0].final_history is not many[1].final_history
    assert not np.shares_memory(many[0].final_history.chosen, many[1].final_history.chosen)


def test_simulation_features_match_replayed_histories():
    for reset in (False, True):
        result = simulate_history(scaffold(), alpha=1, seed=2, reset_history_each_session=reset)
        frame = causal_history(result.choices, reset_history_each_session=reset)
        # Independent arithmetic from the observed prefix, not generator diagnostics.
        from scipy.special import expit
        u = np.array(DEFAULT_UTILITIES)
        predicted = expit(u[frame.left_stimulus - 1] - u[frame.right_stimulus - 1]
                          + frame.left_history - frame.right_history)
        np.testing.assert_allclose(result.probabilities, predicted, atol=1e-15)


def test_invalid_chronology_or_actions_rejected():
    frame = choices()
    with pytest.raises(ValueError, match="Duplicate"):
        causal_history(pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(ValueError, match="missing"):
        causal_history(frame.assign(synthetic_action=None))


def test_frozen_probes_do_not_update_history_or_hardcode_favored_member():
    from reward_pairs.history_evaluation import equal_value_probes, schedule_expected_rates
    from reward_pairs.history_simulation import HistoryRollout
    # A different opponent schedule favors 2 over 3: the evaluator must discover it.
    t = [Trial("p", 1, 1, 2, 1, None), Trial("p", 1, 2, 3, 8, None)]
    expected = schedule_expected_rates(t)
    assert expected[1] > expected[2]
    h = ChoiceHistory()
    for _ in range(6):
        h.update(2, 1, "left")
        h.update(3, 8, "right")
    before = h.chosen.copy(), h.presented.copy()
    rollout = HistoryRollout(pd.DataFrame(), np.array([]), h)
    positive = equal_value_probes([rollout], alpha=1, beta=1, expected_rates=expected)[0]
    zero = equal_value_probes([rollout], alpha=0, beta=1, expected_rates=expected)[0]
    assert positive["expected_stronger"] == positive["stronger_history"] == 2
    assert positive["probability_left"] > 0.5
    assert zero["probability_left"] == zero["probability_right"] == 0.5
    np.testing.assert_array_equal(before[0], h.chosen)
    np.testing.assert_array_equal(before[1], h.presented)
