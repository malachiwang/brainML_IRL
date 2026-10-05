"""Unregularized sequential conditional likelihood; unrestricted history strength."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.special import expit

from .history import causal_history
from .recovery import RecoveryError, RecoveryResult


@dataclass
class HistoryRecoveryResult(RecoveryResult):
    alpha: float
    information_min_eigenvalue: float
    information_condition: float
    optimizer_iterations: int
    reset_history_each_session: bool


def history_design(frame: pd.DataFrame, beta: float) -> np.ndarray:
    """Seven reference-constrained utility differences and one history contrast."""
    pairs = frame[["left_stimulus", "right_stimulus"]].to_numpy(dtype=int) - 1
    stimulus = np.eye(8)[pairs[:, 0]] - np.eye(8)[pairs[:, 1]]
    return beta * np.column_stack([stimulus[:, 1:], frame.left_history - frame.right_history])


def check_identification(design: np.ndarray, labels: np.ndarray) -> None:
    """Reject deficient rank and complete/quasi-separation of the full design.

    Find a bounded direction d with all signed margins >=0 and positive mean.
    Such a direction improves likelihood without a finite maximum. Bounds
    normalize this diagnostic direction; they DO NOT constrain fitted alpha.
    """
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise RecoveryError("Unidentified history model: rank-deficient design (disconnected stimuli or confounded history).")
    signed = (2 * labels - 1)[:, None] * design
    check = linprog(-signed.mean(axis=0), A_ub=-signed, b_ub=np.zeros(len(signed)),
                    bounds=[(-1, 1)] * design.shape[1], method="highs")
    if not check.success:
        raise RecoveryError(f"Separation diagnostic failed: {check.message}")
    if -check.fun > 1e-8:
        raise RecoveryError("Separated choices: no finite unregularized history MLE.")


def recover_history(
    choices: pd.DataFrame, *, beta: float = 1.0,
    fit_sessions: Sequence[int] | None = None, reset_history_each_session: bool = False,
    max_iterations: int = 1000,
) -> HistoryRecoveryResult:
    """Reconstruct observed pre-action history, then fit w_2..w_8 and free alpha.

    No true parameters, history labels, or reward columns are accepted as
    model inputs. fit_sessions selects likelihood rows AFTER causal replay.
    Observed histories are fixed covariates during optimization, not simulated
    again at candidate parameters. No derivative through observed actions.
    """
    if not np.isfinite(beta) or beta <= 0 or max_iterations < 1:
        raise ValueError("Require finite beta > 0 and a positive iteration limit.")
    frame = causal_history(choices, reset_history_each_session=reset_history_each_session)
    if fit_sessions is not None:
        frame = frame.loc[frame.session.isin(fit_sessions)]
    if frame.empty:
        raise ValueError("No trials in requested fitting sessions.")
    design = history_design(frame, beta)
    labels = frame.synthetic_action.eq("left").to_numpy(dtype=float)
    check_identification(design, labels)
    signed = (2 * labels - 1)[:, None] * design

    def objective(parameters):
        margins = signed @ parameters
        return float(np.logaddexp(0, -margins).mean()), -signed.T @ expit(-margins) / len(margins)

    fit = minimize(objective, np.zeros(8), jac=True, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": max_iterations})
    if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise RecoveryError(f"History optimization failed: {fit.message}")
    logits = design @ fit.x
    information = design.T @ ((expit(logits) * expit(-logits))[:, None] * design)
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 0 or not np.isfinite(eigenvalues).all():
        raise RecoveryError("Singular/nonfinite observed information at the history optimum.")
    return HistoryRecoveryResult(
        utilities=pd.Series(np.r_[0, fit.x[:7]], index=pd.Index(range(1, 9), name="stimulus"), name="recovered"),
        beta=beta, reference=1, negative_log_likelihood=float(fit.fun * len(frame)), n_trials=len(frame),
        alpha=float(fit.x[7]), information_min_eigenvalue=float(eigenvalues[0]),
        information_condition=float(eigenvalues[-1] / eigenvalues[0]), optimizer_iterations=int(fit.nit),
        reset_history_each_session=reset_history_each_session,
    )


def predict_history(
    result: RecoveryResult, choices: pd.DataFrame, *, sessions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """One-step predictions: pass full prefixes, score before observing each action.

    A plain RecoveryResult is the existing static model (alpha=0). For history
    fits, the training reset convention is automatically reused at prediction.
    The returned chronology makes every reported probability auditable.
    """
    reset = getattr(result, "reset_history_each_session", False)
    frame = causal_history(choices, reset_history_each_session=reset)
    parameters = np.r_[result.utilities.reindex(range(2, 9)).to_numpy(), getattr(result, "alpha", 0.0)]
    logits = history_design(frame, result.beta) @ parameters
    signed = np.where(frame.synthetic_action.eq("left"), 1, -1)
    frame["probability_left"] = expit(logits)
    frame["log_loss"] = np.logaddexp(0, -signed * logits)
    return frame if sessions is None else frame.loc[frame.session.isin(sessions)].copy()
