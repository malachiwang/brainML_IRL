from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reward_pairs.history_recovery import check_identification, predict_history, recover_history
from reward_pairs.history_simulation import independent_rollouts
from reward_pairs.recovery import RecoveryError, recover_utilities
from reward_pairs.trajectories import Trial


@pytest.fixture(scope="module")
def controlled_scaffold():
    # Many independent short agents keep early, informative history variation.
    pairs = np.array(list(combinations(range(1, 9), 2)) * 5)
    rng = np.random.default_rng(700)
    trials = []
    for session in range(1, 6):
        for i, (left, right) in enumerate(pairs[rng.permutation(len(pairs))]):
            trials.append(Trial("controlled", session, i + 1, int(left), int(right), None))
    return trials


def pooled(scaffold, alpha, seed=30, n=40):
    return pd.concat([r.choices for r in independent_rollouts(scaffold, n, alpha=alpha, seed=seed)], ignore_index=True)


@pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.5, 1.0])
def test_known_positive_negative_and_zero_alpha_recovery(controlled_scaffold, alpha):
    data = pooled(controlled_scaffold, alpha)
    fit = recover_history(data)
    assert fit.utilities.loc[1] == 0
    assert abs(fit.alpha - alpha) < 0.3
    np.testing.assert_allclose(fit.utilities, [0, 1, 1, 2, 2, 3, 3, 4], atol=0.16)
    assert fit.information_min_eigenvalue > 0 and fit.information_condition < 1e4
    if alpha < 0:
        assert fit.alpha < 0


def test_future_changes_leave_training_fit_and_earlier_predictions_unchanged(controlled_scaffold):
    data = pooled(controlled_scaffold, 1, n=5)
    changed = data.copy()
    changed.loc[changed.session == 5, "synthetic_action"] = "right"
    a = recover_history(data, fit_sessions=[1, 2, 3, 4])
    b = recover_history(changed, fit_sessions=[1, 2, 3, 4])
    np.testing.assert_array_equal(a.utilities, b.utilities)
    assert a.alpha == b.alpha
    prefix = recover_history(data[data.session < 5])
    np.testing.assert_array_equal(a.utilities, prefix.utilities)
    pa, pb = predict_history(a, data), predict_history(a, changed)
    # First test action itself cannot change its probability, either.
    before_or_first_test = (pa.session < 5) | ((pa.session == 5) & (pa.trial == 1))
    np.testing.assert_array_equal(pa.loc[before_or_first_test, "probability_left"], pb.loc[before_or_first_test, "probability_left"])
    # Session 5 inherits the actual session-4 counts, not an artificial reset.
    first_test = pa[(pa.rollout == 0) & (pa.session == 5)].iloc[0]
    assert abs(first_test.left_history) + abs(first_test.right_history) > 0


def test_no_truth_or_supplied_history_leakage(controlled_scaffold):
    data = pooled(controlled_scaffold, 0.5, n=5)
    a = recover_history(data)
    b = recover_history(data.assign(alpha_true=-100, reward1=999, left_history=1e6, right_history=-1e6))
    np.testing.assert_array_equal(a.utilities, b.utilities)
    assert a.alpha == b.alpha


def test_design_and_stable_likelihood_agree(controlled_scaffold):
    data = pooled(controlled_scaffold, 1, n=5)
    fit = recover_history(data)
    prediction = predict_history(fit, data)
    assert abs(prediction.log_loss.sum() - fit.negative_log_likelihood) < 1e-8
    # Setting alpha to zero reduces this design to the existing BT predictor.
    static = recover_utilities(data)
    assert fit.negative_log_likelihood <= static.negative_log_likelihood + 1e-8
    from reward_pairs.simulation import left_probability
    expected = left_probability(prediction[["left_stimulus", "right_stimulus"]].to_numpy(), static.utilities)
    np.testing.assert_allclose(predict_history(static, data).probability_left, expected, atol=1e-15)


def test_disconnected_and_history_confounded_cases_rejected():
    data = pd.DataFrame(dict(rollout=[0, 0], session=[1, 1], trial=[1, 2],
                             left_stimulus=[1, 1], right_stimulus=[2, 2], synthetic_action=["left", "right"]))
    with pytest.raises(RecoveryError, match="rank-deficient"):
        recover_history(data)
    # All observations are first trials of separate agents: no history variation,
    # despite a connected comparison graph and both outcomes on every pair.
    rows = [(i, 1, 1, left, right, a) for i, (left, right, a) in enumerate(
        (l, r, a) for l, r in combinations(range(1, 9), 2) for a in ("left", "right"))]
    from reward_pairs.history import CHOICE_COLUMNS
    with pytest.raises(RecoveryError, match="rank-deficient"):
        recover_history(pd.DataFrame(rows, columns=CHOICE_COLUMNS))


def test_full_design_separation_not_just_static_graph():
    # Full rank but every feature supports only a positive response.
    with pytest.raises(RecoveryError, match="Separated"):
        check_identification(np.eye(8), np.ones(8))
    check_identification(np.vstack([np.eye(8), np.eye(8)]), np.r_[np.ones(8), np.zeros(8)])


def test_optimizer_failure_and_nonfinite_output_explicit(controlled_scaffold, monkeypatch):
    data = pooled(controlled_scaffold, 1, n=2)
    with pytest.raises(RecoveryError, match="optimization failed"):
        recover_history(data, max_iterations=1)
    monkeypatch.setattr("reward_pairs.history_recovery.minimize", lambda *a, **k: SimpleNamespace(
        success=True, x=np.full(8, np.nan), fun=1., message="invalid stub"))
    with pytest.raises(RecoveryError, match="optimization failed"):
        recover_history(data)


@pytest.mark.parametrize("alpha", [0., 1.])
def test_average_heldout_prediction_and_zero_control(controlled_scaffold, alpha):
    # Average across independent datasets, not a per-seed monotonic requirement.
    gains = []
    estimates = []
    for seed in [10, 11, 12, 13, 14]:
        data = pooled(controlled_scaffold, alpha, seed=seed, n=20)
        h = recover_history(data, fit_sessions=[1, 2, 3, 4])
        b = recover_utilities(data[data.session < 5])
        gains.append(predict_history(b, data, sessions=[5]).log_loss.mean()
                     - predict_history(h, data, sessions=[5]).log_loss.mean())
        estimates.append(h.alpha)
    if alpha > 0:
        assert np.mean(gains) > 0
    else:
        # Finite noise can give either sign; require no meaningful systematic gain.
        assert np.mean(gains) < 5e-4
        assert abs(np.mean(estimates)) < 0.2
