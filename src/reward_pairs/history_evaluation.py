"""Matched-model experiment diagnostics. Synthetic truth is used only here."""

import numpy as np
import pandas as pd

from .evaluation import compare_utilities
from .history import ChoiceHistory
from .history_recovery import predict_history, recover_history
from .history_simulation import HistoryRollout, history_probability
from .recovery import RecoveryError, recover_utilities
from .simulation import DEFAULT_UTILITIES, left_probability

EQUAL_PAIRS = ((2, 3), (4, 5), (6, 7))  # Evaluation answer key, never an estimator constraint.


def compare_history_models(choices, *, alpha_true, beta=1.0, reset_history_each_session=False):
    """Fit identical data twice: all sessions for recovery, 1..4 for prediction."""
    report = {"fits": {}, "status": "ok"}
    fits = {}
    for scope, sessions in (("all", None), ("train", (1, 2, 3, 4))):
        selected = choices if sessions is None else choices.loc[choices.session.isin(sessions)]
        for model in ("static", "history"):
            key = f"{scope}_{model}"
            try:
                fit = (recover_utilities(selected, beta=beta) if model == "static" else
                       recover_history(choices, beta=beta, fit_sessions=sessions,
                                       reset_history_each_session=reset_history_each_session))
                fits[key] = fit
                _, metrics = compare_utilities(fit)
                report["fits"][key] = dict(
                    status="ok", utilities=fit.utilities.tolist(), alpha=getattr(fit, "alpha", 0.0),
                    alpha_absolute_error=abs(getattr(fit, "alpha", 0.0) - alpha_true),
                    n_trials=fit.n_trials, nll=fit.negative_log_likelihood, **metrics,
                )
                if model == "history":
                    report["fits"][key].update(information_condition=fit.information_condition,
                                               information_min_eigenvalue=fit.information_min_eigenvalue,
                                               severe_conditioning=fit.information_condition > 1e8)
                if scope == "train":
                    prediction = predict_history(fit, choices, sessions=[5])
                    if prediction.empty:
                        raise ValueError("Held-out comparison requires session 5.")
                    report["fits"][key].update(test_nll=float(prediction.log_loss.sum()),
                                               test_log_loss=float(prediction.log_loss.mean()),
                                               test_trials=len(prediction))
            except RecoveryError as exc:
                report["status"] = "FAILED"
                report["fits"][key] = {"status": str(exc)}
    if all(key in fits for key in ("all_static", "all_history")):
        ps = predict_history(fits["all_static"], choices)
        ph = predict_history(fits["all_history"], choices)
        delta = np.abs(ps.probability_left - ph.probability_left)
        report.update(mean_probability_difference=float(delta.mean()), max_probability_difference=float(delta.max()))
    if all(key in fits for key in ("train_static", "train_history")):
        report["heldout_log_loss_gain"] = (report["fits"]["train_static"]["test_log_loss"]
                                          - report["fits"]["train_history"]["test_log_loss"])
    return report, fits


def schedule_expected_rates(trajectory, *, beta=1.0):
    """Reward-only oracle expectation over the scaffold, for probe direction only.

    This evaluation-only comparator derives the favored member from opponents
    and presentation counts. No stimulus is hard-coded as a history 'winner'.
    """
    pairs = np.array([t.state for t in trajectory])
    probabilities = left_probability(pairs, DEFAULT_UTILITIES, beta=beta)
    presentations, selections = np.zeros(8), np.zeros(8)
    for (left, right), p in zip(pairs, probabilities):
        for stimulus in {left, right}:
            presentations[stimulus - 1] += 1
        selections[left - 1] += p
        selections[right - 1] += 1 - p
    return np.divide(selections, presentations, out=np.full(8, 0.5), where=presentations > 0)


def equal_value_probes(
    rollouts: list[HistoryRollout], *, alpha: float, beta: float, expected_rates: np.ndarray,
    static_fit=None, history_fit=None,
) -> list[dict]:
    """Frozen-history probabilities, not simulated extra trials or human tests."""
    rows = []
    for rollout_id, rollout in enumerate(rollouts):
        history: ChoiceHistory = rollout.final_history
        rates, h = history.rates, history.centered
        for left, right in EQUAL_PAIRS:
            li, ri = left - 1, right - 1
            gap = float(h[li] - h[ri])
            expected_direction = np.sign(expected_rates[li] - expected_rates[ri])
            p = history_probability(left, right, history, DEFAULT_UTILITIES, alpha=alpha, beta=beta)
            row = dict(rollout=rollout_id, left=left, right=right,
                       left_utility=DEFAULT_UTILITIES[li], right_utility=DEFAULT_UTILITIES[ri],
                       left_chosen=int(history.chosen[li]), right_chosen=int(history.chosen[ri]),
                       left_presented=int(history.presented[li]), right_presented=int(history.presented[ri]),
                       left_rate=float(rates[li]), right_rate=float(rates[ri]),
                       left_history=float(h[li]), right_history=float(h[ri]),
                       left_effective=float(DEFAULT_UTILITIES[li] + alpha * h[li]),
                       right_effective=float(DEFAULT_UTILITIES[ri] + alpha * h[ri]),
                       probability_left=p, probability_right=1 - p,
                       stronger_history=int(left if gap > 0 else right) if gap != 0 else None,
                       expected_stronger=int(left if expected_direction > 0 else right) if expected_direction != 0 else None,
                       matches_schedule_direction=bool(np.sign(gap) == expected_direction))
            for label, fit in (("static", static_fit), ("history", history_fit)):
                if fit is not None:
                    row[f"{label}_probability_left"] = history_probability(
                        left, right, history, fit.utilities, alpha=getattr(fit, "alpha", 0.0), beta=beta)
            rows.append(row)
    return rows


def summary_table(runs: list[dict]) -> pd.DataFrame:
    """Keep failed configurations visible with an explicit status."""
    rows = []
    for run in runs:
        row = {k: run[k] for k in ("alpha_true", "seed", "rollouts", "status")}
        for label in ("static", "history"):
            fit = run["fits"][f"all_{label}"]
            if fit["status"] == "ok":
                row[f"{label}_mae"] = fit["mae"]
                if label == "history":
                    row.update(alpha_hat=fit["alpha"], alpha_error=fit["alpha_absolute_error"],
                               condition=fit["information_condition"], order=fit["strict_order_agreement"])
            test = run["fits"][f"train_{label}"]
            if test["status"] == "ok":
                row[f"test_{label}_loss"] = test["test_log_loss"]
        row["heldout_gain"] = run.get("heldout_log_loss_gain", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
