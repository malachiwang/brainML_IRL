"""Unregularized Bradley–Terry maximum likelihood with explicit identification."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from .simulation import stimulus_pairs


class RecoveryError(ValueError):
    """The observed comparisons do not support a finite identified fit."""


@dataclass
class RecoveryResult:
    utilities: pd.Series
    beta: float
    reference: int
    negative_log_likelihood: float
    n_trials: int


def recover_utilities(
    choices: pd.DataFrame, *, beta: float = 1.0, reference: int = 1,
) -> RecoveryResult:
    """Fit from ONLY displayed pairs and synthetic actions; no rewards/truth.

    Fix U_reference = 0 and keep beta known and fixed. Unknown beta and utility
    scale cannot both be identified. Disconnected comparisons or separation
    raise rather than fabricating estimates with an implicit prior/penalty.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Recovery requires a fixed, finite beta > 0.")
    if reference not in range(1, 9):
        raise ValueError("Reference stimulus must be in 1..8.")
    required = {"left_stimulus", "right_stimulus", "synthetic_action"}
    if missing := required - set(choices.columns):
        raise ValueError(f"Missing choice columns: {sorted(missing)}")
    pairs = stimulus_pairs(choices[["left_stimulus", "right_stimulus"]].to_numpy())
    action = choices["synthetic_action"]
    if not action.isin(["left", "right"]).all():
        raise ValueError("Every synthetic action must be left/right; missing choices are not dropped.")
    chose_left = action.eq("left").to_numpy(dtype=bool)
    winner = np.where(chose_left, pairs[:, 0], pairs[:, 1]) - 1
    loser = np.where(chose_left, pairs[:, 1], pairs[:, 0]) - 1
    graph = np.zeros((8, 8), dtype=int)
    graph[winner, loser] = 1
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise RecoveryError("Comparison graph is disconnected: one reference cannot identify all eight utilities.")
    if connected_components(graph, directed=True, connection="strong", return_labels=False) != 1:
        raise RecoveryError("Choices are separated: no finite unregularized MLE. Collect more stochastic choices or lower simulation beta.")

    # Aggregate identical winner/loser observations, preserving their counts.
    # This is the same trial-wise likelihood, with at most 64 distinct rows.
    comparisons, counts = np.unique(np.column_stack([winner, loser]), axis=0, return_counts=True)
    design = np.eye(8)[comparisons[:, 0]] - np.eye(8)[comparisons[:, 1]]
    free = np.arange(8) != reference - 1
    design = beta * design[:, free]
    weights = counts / len(pairs)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        margins = design @ parameters
        loss = float(weights @ np.logaddexp(0.0, -margins))
        gradient = -(design.T @ (weights * expit(-margins)))
        return loss, gradient

    fit = minimize(objective, np.zeros(7), jac=True, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": 1000})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RecoveryError(f"Utility optimization failed: {fit.message}")
    utilities = np.zeros(8)
    utilities[free] = fit.x
    return RecoveryResult(
        pd.Series(utilities, index=pd.Index(range(1, 9), name="stimulus"), name="recovered"),
        beta, reference, float(fit.fun * len(pairs)), len(pairs),
    )
