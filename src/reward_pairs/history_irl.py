"""Unregularized likelihood IRL under exact finite-lookahead soft planning.

Unlike the myopic regression, this objective need not be globally convex.
Convergence and local information checks are not a global-optimum certificate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .history import causal_history
from .history_planning import PlanningBatch, check_planning
from .recovery import RecoveryError, RecoveryResult


def demonstration_states(mdp, choices):
    """Replay only observed past choices. Future *pairs*, never actions, are known.

    Each rollout must supply a complete prefix of the retained scaffold. Loss
    subsets are selected after replay; missing rows cannot silently change time.
    """
    frame = causal_history(choices)
    times, counts = [], []
    for _, group in frame.groupby("rollout", sort=False):
        n = len(group)
        if n > len(mdp.pairs) or tuple(zip(group.session, group.trial)) != mdp.chronology[:n]:
            raise ValueError("Each demonstration must be a complete chronological scaffold prefix.")
        pairs = group[["left_stimulus", "right_stimulus"]].to_numpy(dtype=int)
        if not np.array_equal(pairs, mdp.pairs[:n]):
            raise ValueError("Demonstration displayed pairs differ from known scaffold.")
        selected = np.where(group.synthetic_action.eq("left"), pairs[:, 0], pairs[:, 1]) - 1
        increments = np.eye(8, dtype=int)[selected]
        counts.append(np.vstack([np.zeros(8, dtype=int), increments.cumsum(axis=0)[:-1]]) + mdp.initial.chosen)
        times.extend(range(n))
    return frame, np.asarray(times), np.concatenate(counts)


class PlanningLikelihood:
    """Compile observed roots and hypothetical branches without parameter truth."""

    def __init__(self, mdp, choices, *, beta=1., gamma=.95, lookahead=3, fit_sessions=None, chunk_size=2048):
        check_planning(beta, gamma, lookahead)
        frame, times, counts = demonstration_states(mdp, choices)
        mask = np.ones(len(frame), dtype=bool) if fit_sessions is None else frame.session.isin(fit_sessions).to_numpy()
        if not mask.any() or chunk_size < 1:
            raise ValueError("Need nonempty fitting rows and a positive chunk size.")
        self.frame = frame.loc[mask].reset_index(drop=True)
        self.actions = self.frame.synthetic_action.eq("right").to_numpy(dtype=int)
        self.beta, self.gamma, self.lookahead = beta, gamma, lookahead
        times, counts = times[mask], counts[mask]
        safe_chunk = min(chunk_size, 2_000_000 // (2 ** (lookahead + 1) - 1))
        self.batches, self.batch_indices = [], []
        # Group by remaining scaffold length to avoid padding short toy episodes
        # up to the longest root window. Restore original row order on output.
        available = np.minimum(lookahead, len(mdp.pairs) - 1 - times)
        for depth in np.unique(available):
            indices = np.flatnonzero(available == depth)
            for start in range(0, len(indices), safe_chunk):
                selected = indices[start:start + safe_chunk]
                self.batch_indices.append(selected)
                self.batches.append(PlanningBatch(mdp, times[selected], counts[selected], lookahead=lookahead))

    def probabilities(self, theta, *, gradient=False):
        lp = np.empty((len(self.actions), 2))
        jac = np.empty((len(self.actions), 2, 8)) if gradient else None
        for b, indices in zip(self.batches, self.batch_indices):
            _, lp[indices], derivatives = b.evaluate(theta, beta=self.beta, gamma=self.gamma, gradient=gradient)
            if gradient:
                jac[indices] = derivatives
        return lp, jac

    def objective(self, theta):
        lp, jac = self.probabilities(theta, gradient=True)
        rows = np.arange(len(self.actions))
        return float(-lp[rows, self.actions].mean()), -jac[rows, self.actions].mean(axis=0)


@dataclass
class PlanningRecoveryResult(RecoveryResult):
    alpha: float
    gamma: float
    lookahead: int
    optimizer_iterations: int
    optimizer_evaluations: int
    information_min_eigenvalue: float
    information_condition: float
    gradient_max_abs: float


def recover_planning(mdp, choices, *, beta=1., gamma=.95, lookahead=3,
                     fit_sessions=None, max_iterations=1000):
    if max_iterations < 1:
        raise ValueError("Iteration limit must be positive.")
    problem = PlanningLikelihood(mdp, choices, beta=beta, gamma=gamma, lookahead=lookahead, fit_sessions=fit_sessions)
    fit = minimize(problem.objective, np.zeros(8), jac=True, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": max_iterations})
    if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise RecoveryError(f"Planning optimization failed: {fit.message}")
    # Numerical derivative of the analytic gradient: observed, not expected,
    # information. No old linear-logistic separation theorem is assumed here.
    step = 1e-4
    hessian = np.column_stack([(problem.objective(fit.x + d)[1] - problem.objective(fit.x - d)[1]) / (2 * step)
                               for d in step * np.eye(8)]) * len(problem.actions)
    hessian = (hessian + hessian.T) / 2
    eig = np.linalg.eigvalsh(hessian)
    if not np.isfinite(eig).all() or eig[0] <= max(1e-10, eig[-1] * 1e-10):
        raise RecoveryError("Unidentified/non-minimum planning fit: singular or nonpositive observed information.")
    return PlanningRecoveryResult(
        utilities=pd.Series(np.r_[0., fit.x[:7]], index=pd.Index(range(1, 9), name="stimulus"), name="recovered"),
        beta=beta, reference=1, negative_log_likelihood=float(fit.fun * len(problem.actions)), n_trials=len(problem.actions),
        alpha=float(fit.x[7]), gamma=gamma, lookahead=lookahead, optimizer_iterations=int(fit.nit),
        optimizer_evaluations=int(fit.nfev), information_min_eigenvalue=float(eig[0]),
        information_condition=float(eig[-1] / eig[0]), gradient_max_abs=float(np.max(np.abs(fit.jac))))


def predict_planning(mdp, fit, choices, *, sessions=None):
    problem = PlanningLikelihood(mdp, choices, beta=fit.beta, gamma=fit.gamma, lookahead=fit.lookahead, fit_sessions=sessions)
    lp, _ = problem.probabilities(np.r_[fit.utilities.iloc[1:].to_numpy(), fit.alpha])
    frame = problem.frame.copy()
    frame["probability_left"] = np.exp(lp[:, 0])
    frame["log_loss"] = -lp[np.arange(len(frame)), problem.actions]
    return frame
